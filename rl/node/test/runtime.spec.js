import {expect} from 'chai';
import {createRuntime} from '../runtime';
import {getGameRuntime, gameRandom, ASYNC_GAME_RUNTIME} from '~/shared/utils/gameRuntime';

const draw = (runtime, count = 20) => {
  const values = [];
  for (let index = 0; index < count; index++) {
    values.push(runtime.random());
  }
  return values;
};

describe('rl/node/runtime', () => {
  describe('seed', () => {
    it('accepts zero', () => {
      const runtime = createRuntime({seed: 0});
      expect(runtime.seed).to.equal(0);
      expect(runtime.random()).to.be.a('number');
    });

    it('rejects missing and invalid seeds', () => {
      [undefined, null, -1, 1.5, '1', NaN, Infinity, 4294967296].forEach((seed) => {
        expect(() => createRuntime({seed}), String(seed)).to.throw('INVALID_SEED');
      });
      expect(() => createRuntime({})).to.throw('INVALID_SEED');
      expect(() => createRuntime()).to.throw('INVALID_SEED');
    });
  });

  describe('random', () => {
    it('produces identical sequences for identical seeds', () => {
      expect(draw(createRuntime({seed: 123}))).to.deep.equal(draw(createRuntime({seed: 123})));
    });

    it('produces different sequences for different seeds', () => {
      expect(draw(createRuntime({seed: 123}))).to.not.deep.equal(draw(createRuntime({seed: 456})));
    });

    it('stays within [0, 1)', () => {
      const runtime = createRuntime({seed: 7});
      for (let index = 0; index < 1000; index++) {
        const value = runtime.random();
        expect(value).to.be.at.least(0);
        expect(value).to.be.below(1);
      }
    });

    it('keeps interleaved runtimes independent', () => {
      const left = createRuntime({seed: 123});
      const right = createRuntime({seed: 123});
      const noise = createRuntime({seed: 456});
      const leftValues = [];
      const rightValues = [];
      for (let index = 0; index < 10; index++) {
        leftValues.push(left.random());
        noise.random();
        rightValues.push(right.random());
        noise.random();
      }
      expect(rightValues).to.deep.equal(leftValues);
    });
  });

  describe('ids', () => {
    it('is monotone and does not consume the RNG', () => {
      const plain = createRuntime({seed: 42});
      const withIds = createRuntime({seed: 42});
      expect(withIds.nextId('animal')).to.equal('animal_1');
      expect(withIds.nextId('animal')).to.equal('animal_2');
      expect(withIds.id('card')).to.equal('card_3');
      expect(draw(withIds)).to.deep.equal(draw(plain));
    });

    it('never re-issues reserved ids', () => {
      const runtime = createRuntime({seed: 1});
      runtime.reserve('animal_1');
      runtime.reserve('A_5');
      expect(runtime.hasId('A_5')).to.equal(true);
      expect(runtime.hasId('A_6')).to.equal(false);
      expect(runtime.nextId('animal')).to.equal('animal_2');
      expect(runtime.nextId('animal')).to.equal('animal_3');
    });
  });

  describe('clock', () => {
    it('starts at zero and ticks by one', () => {
      const runtime = createRuntime({seed: 0});
      expect(runtime.now()).to.equal(0);
      runtime.tick();
      expect(runtime.now()).to.equal(1);
      runtime.tick();
      runtime.tick();
      expect(runtime.now()).to.equal(3);
    });
  });

  describe('capture/restore', () => {
    it('round-trips through JSON and replays rng, ids and clock', () => {
      const runtime = createRuntime({seed: 99});
      runtime.random();
      runtime.tick();
      runtime.nextId('animal');
      runtime.reserve('forced_7');
      const captured = JSON.parse(JSON.stringify(runtime.capture()));
      expect(captured).to.deep.equal({
        seed: 99,
        rngState: runtime.capture().rngState,
        idCounter: 1,
        clock: 1,
        reserved: ['forced_7'],
      });

      const firstRun = draw(runtime, 5);
      const firstId = runtime.nextId('animal');
      runtime.restore(captured);
      expect(runtime.now()).to.equal(1);
      expect(runtime.hasId('forced_7')).to.equal(true);
      expect(draw(runtime, 5)).to.deep.equal(firstRun);
      expect(runtime.nextId('animal')).to.equal(firstId);
    });

    it('rejects malformed state', () => {
      const runtime = createRuntime({seed: 1});
      [null, 5, [], {}, {seed: 1}, {...runtime.capture(), idCounter: -1}, {...runtime.capture(), reserved: 'x'}]
        .forEach((state) => {
          expect(() => runtime.restore(state)).to.throw('INVALID_RUNTIME_STATE');
        });
    });
  });

  describe('withGameRuntime integration', () => {
    it('activates the runtime for the duration of run()', () => {
      const runtime = createRuntime({seed: 5});
      const mirror = createRuntime({seed: 5});
      expect(getGameRuntime()).to.equal(null);
      const result = runtime.run(() => {
        expect(getGameRuntime()).to.equal(runtime);
        expect(gameRandom()).to.equal(mirror.random());
        return 'ok';
      });
      expect(result).to.equal('ok');
      expect(getGameRuntime()).to.equal(null);
    });

    it('restores the previous context when nested', () => {
      const outer = createRuntime({seed: 1});
      const inner = createRuntime({seed: 2});
      outer.run(() => {
        inner.run(() => {
          expect(getGameRuntime()).to.equal(inner);
        });
        expect(getGameRuntime()).to.equal(outer);
      });
    });

    it('rejects async callbacks', () => {
      const runtime = createRuntime({seed: 5});
      expect(() => runtime.run(() => Promise.resolve())).to.throw(ASYNC_GAME_RUNTIME);
      expect(getGameRuntime()).to.equal(null);
    });
  });
});
