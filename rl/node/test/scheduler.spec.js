import {expect} from 'chai';
import {createScheduler} from '../scheduler';

const ADD = '@@reduxTimeoutMiddleware/addTimeout';
const CANCEL = '@@reduxTimeoutMiddleware/cancelTimeout';
const CHECK = '@@reduxTimeoutMiddleware/checkTimeout';
const SHIFT = '@@reduxTimeoutMiddleware/testShiftTime';

const add = (duration, name, callback) => ({type: ADD, data: {duration, name, callback}});
const cancel = (name) => ({type: CANCEL, data: {name}});
const check = (name) => ({type: CHECK, data: {name}});
const shift = (time) => ({type: SHIFT, data: {time}});

// Thunk-like tail: functions are invoked, plain actions are recorded.
const setup = () => {
  const scheduler = createScheduler();
  const passed = [];
  let chain = null;
  const store = {
    getState: () => ({}),
    dispatch: (action) => chain(action),
  };
  chain = scheduler.middleware(store)((action) => {
    if (typeof action === 'function') {
      return action(store.dispatch, store.getState);
    }
    passed.push(action);
    return action;
  });
  return {scheduler, dispatch: store.dispatch, passed};
};

describe('rl/node/scheduler', () => {
  it('arms, reports and cancels a timeout', () => {
    const {dispatch, passed} = setup();
    dispatch(add(1000, 'turn', () => dispatch({type: 'TICK'})));

    expect(dispatch(check('turn'))).to.equal(true);
    expect(dispatch(cancel('turn'))).to.equal(1000);
    expect(dispatch(check('turn'))).to.equal(false);
    expect(dispatch(cancel('turn'))).to.equal(undefined);
    expect(passed).to.deep.equal([]);
  });

  it('does not schedule zero-duration timeouts', () => {
    const {scheduler, dispatch} = setup();
    expect(dispatch(add(0, 'turn', () => null))).to.equal(undefined);
    expect(dispatch(check('turn'))).to.equal(false);
    expect(scheduler.pending()).to.deep.equal([]);
  });

  it('throws on a duplicate name', () => {
    const {dispatch} = setup();
    dispatch(add(10, 'turn', () => null));
    expect(() => dispatch(add(10, 'turn', () => null))).to.throw('timeout[turn] already occupied');
  });

  it('shifts remaining time without firing', () => {
    const {scheduler, dispatch, passed} = setup();
    dispatch(add(1000, 'turn', () => dispatch({type: 'TICK'})));
    dispatch(add(500, 'question', () => dispatch({type: 'QUESTION'})));

    dispatch(shift(400));
    expect(scheduler.pending()).to.deep.equal([
      {name: 'turn', remaining: 600, order: 0},
      {name: 'question', remaining: 100, order: 1},
    ]);

    dispatch(shift(10000));
    expect(scheduler.pending().map(({name, remaining}) => ({name, remaining}))).to.deep.equal([
      {name: 'turn', remaining: 0},
      {name: 'question', remaining: 0},
    ]);

    expect(passed).to.deep.equal([]);
    expect(dispatch(check('turn'))).to.equal(true);
  });

  it('fires a callback exactly once', () => {
    const {scheduler, dispatch, passed} = setup();
    dispatch(add(10, 'turn', () => dispatch({type: 'TICK'})));

    expect(scheduler.fire('turn')).to.equal(true);
    expect(scheduler.fire('turn')).to.equal(false);
    expect(passed).to.deep.equal([{type: 'TICK'}]);
    expect(dispatch(check('turn'))).to.equal(false);
    expect(scheduler.fire('missing')).to.equal(false);
  });

  it('re-entrant fire cannot run the callback twice', () => {
    const {scheduler, dispatch, passed} = setup();
    let nested = null;
    dispatch(add(10, 'turn', () => {
      nested = scheduler.fire('turn');
      dispatch({type: 'TICK'});
    }));

    expect(scheduler.fire('turn')).to.equal(true);
    expect(nested).to.equal(false);
    expect(passed).to.deep.equal([{type: 'TICK'}]);
    expect(scheduler.pending()).to.deep.equal([]);
  });

  it('captures and restores entries with callable callbacks', () => {
    const {scheduler, dispatch, passed} = setup();
    const first = () => dispatch({type: 'FIRST'});
    const second = () => dispatch({type: 'SECOND'});
    dispatch(add(1000, 'turn', first));
    dispatch(add(500, 'question', second));
    dispatch(shift(250));

    const captured = scheduler.capture();
    expect(captured.entries).to.deep.equal([
      {name: 'turn', remaining: 750, order: 0},
      {name: 'question', remaining: 250, order: 1},
    ]);
    expect(captured.callbacks.map(({name}) => name)).to.deep.equal(['turn', 'question']);

    scheduler.fire('turn');
    expect(passed).to.deep.equal([{type: 'FIRST'}]);
    expect(scheduler.pending()).to.deep.equal([{name: 'question', remaining: 250, order: 1}]);

    scheduler.restore(captured);
    expect(scheduler.pending()).to.deep.equal(captured.entries);
    // Restoring the same handle twice is allowed and idempotent.
    scheduler.restore(captured);
    expect(scheduler.pending()).to.deep.equal(captured.entries);

    expect(scheduler.fire('question')).to.equal(true);
    expect(scheduler.fire('question')).to.equal(false);
    expect(scheduler.fire('turn')).to.equal(true);
    expect(scheduler.fire('turn')).to.equal(false);
    expect(passed).to.deep.equal([{type: 'FIRST'}, {type: 'SECOND'}, {type: 'FIRST'}]);
  });

  it('rejects malformed captured state', () => {
    const {scheduler} = setup();
    [null, 1, {}, {entries: [], callbacks: []}, {entries: {}, callbacks: [], nextOrder: 0},
      {entries: [{name: 'x', remaining: 1, order: 0}], callbacks: [], nextOrder: 1},
      {entries: [{name: 'x', remaining: 1, order: 0}], callbacks: [{name: 'x', callback: 'nope'}], nextOrder: 1}]
      .forEach((state) => {
        expect(() => scheduler.restore(state)).to.throw('INVALID_SCHEDULER_STATE');
      });
  });

  it('passes unrelated actions to next', () => {
    const {dispatch, passed} = setup();
    const action = {type: 'gameEndTurnRequest', data: {}};
    expect(dispatch(action)).to.equal(action);
    expect(passed).to.deep.equal([action]);
  });

  it('resets everything', () => {
    const {scheduler, dispatch} = setup();
    dispatch(add(1000, 'turn', () => dispatch({type: 'TICK'})));
    scheduler.reset();
    expect(scheduler.pending()).to.deep.equal([]);
    expect(dispatch(check('turn'))).to.equal(false);
    expect(scheduler.fire('turn')).to.equal(false);
    dispatch(add(10, 'turn', () => dispatch({type: 'TICK'})));
    expect(scheduler.pending()).to.deep.equal([{name: 'turn', remaining: 10, order: 0}]);
  });
});
