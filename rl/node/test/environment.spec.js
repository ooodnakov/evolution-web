import {expect} from 'chai';
import {List} from 'immutable';

import {createEnvironment, EnvironmentError} from '../environment';
import {EnvironmentState} from '../store';
import {PHASE} from '~/shared/models/game/GameModel';

const ALL_ADDONS = ['base2', 'timeToFly', 'continents', 'bonus', 'plantarium', 'customff', 'lifecycle'];

const lcg = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
};

const drive = (env, frame, limit, random = lcg(1), record = null) => {
  let steps = 0;
  while (frame.actor && steps < limit) {
    const view = frame.observations[frame.actor];
    const index = Math.floor(random() * view.choices.length);
    if (record) record.push(index);
    frame = env.step({decisionId: frame.decisionId, choiceIndex: index});
    steps += 1;
  }
  return {frame, steps};
};

const playOut = (env, seed, limit = 20000) => {
  let frame = env.reset({seed});
  const result = drive(env, frame, limit, lcg(seed + 1));
  return {frame: result.frame, steps: result.steps, hash: env.stateHash()};
};

const codes = (fn) => {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error('expected a thrown environment error');
};

describe('environment', function environmentSuite() {
  this.timeout(120000);
  describe('natural terminal', () => {
    it('finishes a base 2p game with real state transitions and zero-sum rewards', () => {
      const env = createEnvironment({numPlayers: 2});
      let frame = env.reset({seed: 0});
      let sawAnimals = false;
      let steps = 0;
      const random = lcg(99);
      while (frame.actor && steps < 20000) {
        if (frame.observations.player_0.animals.length > 0) sawAnimals = true;
        const view = frame.observations[frame.actor];
        frame = env.step({
          decisionId: frame.decisionId
          , choiceIndex: Math.floor(random() * view.choices.length)
        });
        steps += 1;
      }

      expect(sawAnimals, 'animals were deployed').to.equal(true);
      expect(frame.actor).to.equal(null);
      expect(frame.terminated.player_0).to.equal(true);
      expect(frame.truncated.player_0).to.equal(false);
      expect(frame.info.player_0.outcome).to.have.property('winner');
      const game = env.game();
      expect(game.status.phase).to.equal(PHASE.FINAL);
      expect(game.winnerId).to.equal(frame.info.player_0.outcome.winner);
      const sum = frame.rewards.player_0 + frame.rewards.player_1;
      expect(sum).to.be.closeTo(0, 1e-9);
      expect(frame.rewards.player_0).to.be.oneOf([1, -1]);
      env.close();
    });
  });

  describe('determinism and isolation', () => {
    it('replays the same seed and choices to the same state hash', () => {
      const first = createEnvironment({numPlayers: 2});
      const second = createEnvironment({numPlayers: 2});
      const a = playOut(first, 123);
      const b = playOut(second, 123);
      expect(a.steps).to.equal(b.steps);
      expect(a.hash).to.equal(b.hash);
      expect(a.frame.rewards).to.deep.equal(b.frame.rewards);
      first.close();
      second.close();
    });

    it('does not perturb an episode when another env is interleaved', () => {
      const solo = playOut(createEnvironment({numPlayers: 2}), 123);

      const left = createEnvironment({numPlayers: 2});
      const right = createEnvironment({numPlayers: 2});
      let leftFrame = left.reset({seed: 123});
      let rightFrame = right.reset({seed: 456});
      const leftRandom = lcg(124);
      const rightRandom = lcg(457);
      let steps = 0;
      while (leftFrame.actor && steps < 20000) {
        const view = leftFrame.observations[leftFrame.actor];
        leftFrame = left.step({
          decisionId: leftFrame.decisionId
          , choiceIndex: Math.floor(leftRandom() * view.choices.length)
        });
        if (rightFrame.actor) {
          const rightView = rightFrame.observations[rightFrame.actor];
          rightFrame = right.step({
            decisionId: rightFrame.decisionId
            , choiceIndex: Math.floor(rightRandom() * rightView.choices.length)
          });
        }
        steps += 1;
      }
      expect(steps).to.equal(solo.steps);
      expect(left.stateHash()).to.equal(solo.hash);
      left.close();
      right.close();
    });
  });

  describe('reset', () => {
    it('keeps a single game and a non-playing room across many resets', () => {
      const env = createEnvironment({numPlayers: 3});
      for (let index = 0; index < 100; index += 1) {
        env.reset({seed: index});
        const state = env.store.getState();
        expect(state).to.be.instanceOf(EnvironmentState);
        expect(state.get('games').size).to.equal(1);
        const room = state.getIn(['rooms', 'room_env']);
        expect(room.gameId).to.equal(null);
        expect(room.users.size).to.equal(3);
      }
      env.close();
      env.close();
    });
  });

  describe('snapshot and restore', () => {
    it('restores the exact branch and rejects foreign handles', () => {
      const env = createEnvironment({numPlayers: 2});
      let frame = env.reset({seed: 11});
      const warmup = drive(env, frame, 120, lcg(7));
      frame = warmup.frame;
      const handle = env.snapshot();
      const baseline = env.stateHash();

      const branchChoices = [];
      const branch = drive(env, frame, 25, lcg(8), branchChoices);
      const branchHash = env.stateHash();

      frame = env.restore(handle);
      expect(env.stateHash()).to.equal(baseline);
      const restoredDecision = frame.decisionId;

      const replayChoices = [];
      const replay = drive(env, frame, 25, lcg(8), replayChoices);
      expect(replayChoices).to.deep.equal(branchChoices);
      expect(env.stateHash()).to.equal(branchHash);

      const other = createEnvironment({numPlayers: 2});
      other.reset({seed: 11});
      expect(codes(() => other.restore(handle)).code).to.equal('INVALID_SNAPSHOT');
      expect(restoredDecision).to.be.a('number');
      env.close();
      other.close();
    });

    it('issues a fresh decision token without changing choices', () => {
      const env = createEnvironment({numPlayers: 2});
      const frame = env.reset({seed: 3});
      const handle = env.snapshot();
      const restored = env.restore(handle);
      expect(restored.decisionId).to.be.greaterThan(frame.decisionId);
      expect(restored.observations).to.deep.equal(frame.observations);
      env.close();
    });
  });

  describe('protocol and atomicity', () => {
    it('rejects stale tokens, bad indexes and inactive actors without changing state', () => {
      const env = createEnvironment({numPlayers: 2});
      const frame = env.reset({seed: 4});
      const hash = env.stateHash();

      expect(codes(() => env.step({decisionId: frame.decisionId + 1, choiceIndex: 0})).code)
        .to.equal('INVALID_DECISION');
      expect(codes(() => env.step({decisionId: frame.decisionId, choiceIndex: 999})).code)
        .to.equal('INVALID_CHOICE');
      expect(codes(() => env.step({decisionId: frame.decisionId, choiceIndex: -1})).code)
        .to.equal('INVALID_CHOICE');
      expect(codes(() => env.step({decisionId: frame.decisionId, choiceIndex: 1.5})).code)
        .to.equal('INVALID_CHOICE');
      expect(env.stateHash()).to.equal(hash);
      expect(env.decision.id).to.equal(frame.decisionId);
      env.close();
    });

    it('reports truncation instead of a fake winner', () => {
      const env = createEnvironment({numPlayers: 2, maxDecisions: 3});
      let frame = env.reset({seed: 0});
      const random = lcg(5);
      while (frame.actor) {
        const view = frame.observations[frame.actor];
        frame = env.step({
          decisionId: frame.decisionId
          , choiceIndex: Math.floor(random() * view.choices.length)
        });
      }
      expect(frame.truncated.player_0).to.equal(true);
      expect(frame.terminated.player_0).to.equal(false);
      expect(frame.rewards.player_0).to.equal(0);
      expect(frame.info.player_0.truncation_reason).to.equal('max_decisions');
      expect(frame.info.player_0.outcome).to.equal(null);
      env.close();
    });
  });

  describe('replay artifact', () => {
    it('records local-slot commands and canonical checkpoints', () => {
      const env = createEnvironment({numPlayers: 2});
      let frame = env.reset({seed: 8});
      const random = lcg(2);
      for (let index = 0; index < 20 && frame.actor; index += 1) {
        const view = frame.observations[frame.actor];
        frame = env.step({
          decisionId: frame.decisionId
          , choiceIndex: Math.floor(random() * view.choices.length)
        });
      }
      const replay = env.exportReplay();
      expect(replay.schema_version).to.equal(1);
      expect(replay.seed).to.equal(8);
      expect(replay.commands.length).to.be.greaterThan(0);
      expect(replay.checkpoints.length).to.equal(
        replay.commands.filter(command => command.result === 'committed').length
      );
      replay.checkpoints.forEach((checkpoint) => {
        expect(checkpoint.hash).to.match(/^[0-9a-f]{64}$/);
      });
      const wire = JSON.stringify(replay);
      expect(() => JSON.parse(wire)).to.not.throw();
      expect(wire).to.not.contain('undefined');

      const observation = JSON.stringify(frame.observations);
      expect(observation).to.not.contain('"seed"');
      expect(observation).to.not.contain('game_');
      env.close();
    });
  });

  describe('scenarios', () => {
    const gradientScenario = [
      'phase: feeding'
      , 'food: 0'
      , "deck: ''"
      , 'players:'
      , '  - continent: $A TraitCarnivorous, $D TraitSwimming +'
      , '  - continent: $B TraitPoisonous +, $C +, $E TraitSwimming +'
    ].join('\n');

    const attackFixture = (targetId) => {
      const env = createEnvironment({numPlayers: 2, maxDecisions: 200});
      let frame = env.reset({seed: 1, scenario: gradientScenario});
      const pick = (predicate) => {
        const index = env.decision.choices.findIndex(predicate);
        if (index < 0) throw new Error('fixture choice not found');
        frame = env.step({decisionId: env.decision.id, choiceIndex: index});
        return frame;
      };
      pick(choice => choice.type_name === 'traitActivateRequest');
      pick(choice => choice.entity_id === '$A');
      pick(choice => choice.trait_id);
      pick(choice => choice.entity_id === targetId);
      while (frame.actor) {
        frame = env.step({decisionId: frame.decisionId, choiceIndex: 0});
      }
      const game = env.game();
      const result = {
        winner: game.winnerId
        , scores: game.players.valueSeq().map(player => [player.id, player.countScore()]).toJS()
        , reward: frame.rewards.player_0
      };
      env.close();
      return result;
    };

    it('safe hunt and poisonous hunt reach the natural outcomes', () => {
      const safe = attackFixture('$C');
      expect(safe.scores).to.deep.equal([['player_0', 7], ['player_1', 6]]);
      expect(safe.winner).to.equal('player_0');
      expect(safe.reward).to.equal(1);

      const poison = attackFixture('$B');
      expect(poison.scores).to.deep.equal([['player_0', 3], ['player_1', 5]]);
      expect(poison.winner).to.equal('player_1');
      expect(poison.reward).to.equal(-1);
    });

    it('gives every seat a private transient window and applies the chosen vector', () => {
      const scenario = [
        'phase: feeding'
        , 'food: 0'
        , "deck: ''"
        , 'players:'
        , '  - continent: $A TraitAnglerfish, $B +'
        , '  - continent: $C +'
      ].join('\n');
      const env = createEnvironment({numPlayers: 2});
      let frame = env.reset({seed: 5, scenario});
      expect(frame.actor).to.equal('player_0');
      expect(frame.observations.player_0.context.kind).to.equal(5); // TRANSIENT
      expect(frame.observations.player_0.choices.length).to.equal(2);
      expect(frame.observations.player_1.context.kind).to.equal(5);
      expect(frame.observations.player_1.choices.length).to.equal(0);

      frame = env.step({decisionId: frame.decisionId, choiceIndex: 0}); // true
      expect(frame.actor).to.equal('player_1');
      expect(frame.observations.player_1.choices.length).to.equal(1); // READY

      frame = env.step({decisionId: frame.decisionId, choiceIndex: 0});
      expect(frame.actor).to.equal('player_0');
      expect(frame.observations.player_0.context.kind).to.equal(0); // NORMAL

      const anglerfish = env.game().locateAnimal('$A').traits.find(trait => trait.type === 'TraitAnglerfish');
      expect(Boolean(anglerfish.value)).to.equal(true);
      env.close();
    });

    it('parses a linked trait pair and keeps the game playable to terminal', () => {
      const scenario = [
        'phase: feeding'
        , 'food: 3'
        , "deck: ''"
        , 'players:'
        , '  - continent: $A TraitCommunication$B, $B +'
        , '  - continent: $C +'
      ].join('\n');
      const env = createEnvironment({numPlayers: 2});
      let frame = env.reset({seed: 2, scenario});

      const aTrait = env.game().locateAnimal('$A').traits.find(trait => trait.type === 'TraitCommunication');
      const bTrait = env.game().locateAnimal('$B').traits.find(trait => trait.type === 'TraitCommunication');
      expect(aTrait.linkAnimalId).to.equal('$B');
      expect(aTrait.linkId).to.equal(bTrait.id);
      expect(bTrait.linkAnimalId).to.equal('$A');
      expect(bTrait.linkId).to.equal(aTrait.id);

      let steps = 0;
      while (frame.actor && steps < 2000) {
        frame = env.step({decisionId: frame.decisionId, choiceIndex: 0});
        steps += 1;
      }
      expect(frame.terminated.player_0).to.equal(true);
      expect(frame.truncated.player_0).to.equal(false);
      env.close();
    });

    it('rejects malformed scenarios without an active episode', () => {
      const env = createEnvironment({numPlayers: 2, addons: []});
      const expectScenarioError = (scenario) => {
        expect(codes(() => env.reset({seed: 1, scenario})).code).to.equal('INVALID_SCENARIO');
      };
      expectScenarioError('::: not yaml :::');
      expectScenarioError('players:\n  - continent: $A\n  - continent: $B\n  - continent: $C');
      expectScenarioError('players:\n  - continent: $A\n  - continent: $A');
      expectScenarioError('settings:\n  addon_plantarium: true');
      expectScenarioError('food: -1');
      env.close();
    });
  });
});
