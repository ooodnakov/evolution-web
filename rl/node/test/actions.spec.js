import {expect} from 'chai';
import {List, Map} from 'immutable';

import {CTT_PARAMETER, TRAIT_TARGET_TYPE} from '~/shared/models/game/evolution/constants';
import {QuestionRecord} from '~/shared/models/game/GameModel';
import {parseFromRoom} from '~/shared/models/game/GameModel.parse';
import {SettingsRecord} from '~/shared/models/game/GameSettings';
import {validateGameRequest} from '~/shared/actions/game-validation';

import {createRuntime} from '../runtime';
import {createCatalog} from '../catalog';
import {entitiesOfGame} from '../slots';
import {
  COMMAND_TYPES
  , createActionDomain
  , preparationPending
} from '../actions';

const ROOM = {id: 'room_actions_spec', users: List(['p0', 'p1'])};

const buildGame = (yaml, seed = 7) => {
  const runtime = createRuntime({seed});
  return runtime.run(() => parseFromRoom(ROOM, yaml));
};

const makeContext = (game, playerId, decisionKind = 'NORMAL', revision = 1, extra = {}) => ({
  game
  , projection: game.toOthers(playerId)
  , playerId
  , seat: game.getPlayer(playerId).index
  , decisionKind
  , revision
  , ...extra
});

const stable = (value) => {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') {
    const result = {};
    Object.keys(value).sort().forEach((key) => {
      result[key] = stable(value[key]);
    });
    return result;
  }
  return value;
};

const commandKey = (command) => JSON.stringify(stable(command));

const collectLeaves = (domain, context) => {
  const leaves = [];
  const walk = (prefix) => {
    const next = domain.choices(prefix, context);
    if (next.length === 0) {
      if (domain.isComplete(prefix, context)) {
        leaves.push({prefix, command: domain.resolveCommand(prefix, context)});
      }
      return;
    }
    next.forEach((item) => walk([...prefix, item]));
  };
  walk([]);
  return leaves;
};

const requestOf = (command) => (command.special
  ? null
  : {type: command.type, ...command.data});

const validRequest = (context, request) => {
  try {
    validateGameRequest(context.projection, context.playerId, request, 'complete');
    return true;
  } catch (error) {
    if (error && error.userLevelError) return false;
    throw error;
  }
};

const emittedOfType = (leaves, type) => new Set(leaves
  .filter(({command}) => command && command.type === type)
  .map(({command}) => commandKey(command)));

const ownAnimalEntries = (slots, playerId) => slots.animals
  .filter(entry => entry.ownerId === playerId);

const DEPLOY_SEED = `
phase: deploy
deck: 10 camo
players:
  - continent: $A carn, $B
    hand: 2 camo, 2 communication
  - continent: $C, $D
    hand: 2 camo
`;

const FEEDING_SEED = `
settings:
  addon_plantarium: true
phase: feeding
plants: succ $suc +++, leg $leg
players:
  - continent: $A carn, $B, $W wait
    hand: 2 camo
  - continent: $C, $D
    hand: 2 camo
`;

describe('rl/node/actions', () => {
  const catalog = createCatalog();
  const domain = createActionDomain({catalog});

  describe('NORMAL domain', () => {
    it('builds an actual SettingsRecord game inside a runtime', () => {
      const game = buildGame(DEPLOY_SEED);
      expect(game.settings).to.be.instanceOf(SettingsRecord);
      expect(game.status.currentPlayer).to.equal('p0');
      expect(game.id).to.be.a('string');
    });

    it('emits only leaves that validate under public preconditions', () => {
      [DEPLOY_SEED, FEEDING_SEED].forEach((seed, index) => {
        const game = buildGame(seed);
        const context = makeContext(game, 'p0', 'NORMAL', 100 + index);
        const leaves = collectLeaves(domain, context);
        expect(leaves.length, seed).to.be.greaterThan(0);

        leaves.forEach(({prefix, command}) => {
          const request = requestOf(command);
          if (!request) return;
          expect(request.gameId, 'gameId never leaves the engine').to.equal(undefined);
          const stage = preparationPending(request, context) ? 'before_sharing' : 'complete';
          expect(
            () => validateGameRequest(context.projection, context.playerId, request, stage)
            , `${stage}: ${JSON.stringify(stable(request))}`
          ).to.not.throw();
          expect(domain.isAttemptable(prefix, context)).to.equal(1);
        });
      });
    });

    it('reaches every structural leaf of the brute-force product', () => {
      const game = buildGame(DEPLOY_SEED);
      const context = makeContext(game, 'p0', 'NORMAL', 200);
      const slots = entitiesOfGame(context.projection);
      const player = context.projection.getPlayer('p0');
      const leaves = collectLeaves(domain, context);

      // deployAnimal: every own card x insertion position 0..continent.size
      const bruteAnimal = new Set();
      player.hand.forEach((card) => {
        for (let position = 0; position <= player.continent.size; position += 1) {
          const request = {
            type: 'gameDeployAnimalRequest'
            , cardId: card.id
            , animalPosition: position
          };
          if (validRequest(context, request)) {
            bruteAnimal.add(commandKey({type: request.type, data: {
              cardId: card.id
              , animalPosition: position
            }}));
          }
        }
      });
      expect(emittedOfType(leaves, 'gameDeployAnimalRequest')).to.deep.equal(bruteAnimal);

      // deployTrait: every card side x entity x link endpoint
      const bruteTrait = new Set();
      player.hand.forEach((card) => {
        [false, true].forEach((alternateTrait) => {
          const traitData = card.getTraitDataModel(alternateTrait);
          if (!traitData) return;
          let entityEntries = [];
          if (traitData.cardTargetType & CTT_PARAMETER.ANIMAL) entityEntries = slots.animals;
          else if (traitData.cardTargetType & CTT_PARAMETER.PLANT) entityEntries = slots.plants;

          entityEntries.forEach((entityEntry) => {
            const base = {
              type: 'gameDeployTraitRequest'
              , cardId: card.id
              , entityId: entityEntry.id
              , alternateTrait
            };
            if (!traitData.linkTargetType) {
              if (validRequest(context, base)) {
                bruteTrait.add(commandKey({type: base.type, data: {
                  cardId: card.id
                  , entityId: entityEntry.id
                  , alternateTrait
                }}));
              }
              return;
            }
            let linkEntries = [];
            if (traitData.linkTargetType & CTT_PARAMETER.PLANT) linkEntries = slots.plants;
            else if (traitData.linkTargetType & CTT_PARAMETER.ANIMAL) linkEntries = slots.animals;
            linkEntries.forEach((linkEntry) => {
              const request = {...base, linkId: linkEntry.id};
              if (validRequest(context, request)) {
                bruteTrait.add(commandKey({type: request.type, data: {
                  cardId: card.id
                  , entityId: entityEntry.id
                  , alternateTrait
                  , linkId: linkEntry.id
                }}));
              }
            });
          });
        });
      });
      expect(emittedOfType(leaves, 'gameDeployTraitRequest')).to.deep.equal(bruteTrait);
    });

    it('reaches every takeFood and activate leaf of the brute-force product', () => {
      const game = buildGame(FEEDING_SEED);
      const context = makeContext(game, 'p0', 'NORMAL', 201);
      const slots = entitiesOfGame(context.projection);
      const leaves = collectLeaves(domain, context);
      const ownAnimals = ownAnimalEntries(slots, 'p0');

      const bruteFood = new Set();
      ownAnimals.forEach((animalEntry) => {
        if (!context.projection.isPlantarium()) {
          const request = {type: 'traitTakeFoodRequest', animalId: animalEntry.id};
          if (validRequest(context, request)) {
            bruteFood.add(commandKey({type: request.type, data: {animalId: animalEntry.id}}));
          }
          return;
        }
        slots.plants.forEach((plantEntry) => {
          const request = {
            type: 'traitTakeFoodRequest'
            , animalId: animalEntry.id
            , plantId: plantEntry.id
          };
          if (validRequest(context, request)) {
            bruteFood.add(commandKey({type: request.type, data: {
              animalId: animalEntry.id
              , plantId: plantEntry.id
            }}));
          }
        });
      });
      expect(emittedOfType(leaves, 'traitTakeFoodRequest')).to.deep.equal(bruteFood);

      const bruteActivate = new Set();
      ownAnimals.forEach((animalEntry) => {
        animalEntry.animal.traits.toList().forEach((trait) => {
          const traitData = trait.getDataModel();
          let targetSets = [];
          switch (traitData.targetType) {
            case TRAIT_TARGET_TYPE.NONE:
              targetSets = [[]];
              break;
            case TRAIT_TARGET_TYPE.ANIMAL:
              targetSets = slots.animals.map(entry => [entry.id]);
              break;
            case TRAIT_TARGET_TYPE.PLANT:
              targetSets = slots.plants.map(entry => [entry.id]);
              break;
            case TRAIT_TARGET_TYPE.TRAIT:
              targetSets = animalEntry.animal.traits.toList().map(item => [item.id]);
              break;
            case TRAIT_TARGET_TYPE.TWO_TRAITS: {
              const linkedId = animalEntry.id === trait.hostAnimalId
                ? trait.linkAnimalId
                : trait.hostAnimalId;
              const linkedEntry = slots.animalById.get(linkedId);
              if (!linkedEntry) break;
              targetSets = [];
              animalEntry.animal.traits.toList().forEach((first) => {
                linkedEntry.animal.traits.toList().forEach((second) => {
                  targetSets.push([first.id, second.id]);
                });
              });
              break;
            }
            default:
              targetSets = [];
          }
          targetSets.forEach((targets) => {
            const request = {
              type: 'traitActivateRequest'
              , sourceAid: animalEntry.id
              , traitId: trait.id
              , targets
            };
            if (validRequest(context, request)) {
              bruteActivate.add(commandKey({type: request.type, data: {
                sourceAid: animalEntry.id
                , traitId: trait.id
                , targets
              }}));
            }
          });
        });
      });
      expect(emittedOfType(leaves, 'traitActivateRequest')).to.deep.equal(bruteActivate);
    });

    it('keeps the command type order stable', () => {
      const game = buildGame(FEEDING_SEED);
      const context = makeContext(game, 'p0', 'NORMAL', 202);
      const first = domain.rootChoices(context);
      const second = domain.rootChoices(context);
      expect(first).to.deep.equal(second);

      const types = first.filter(item => item.field_kind === 'TYPE').map(item => item.type_name);
      expect(types).to.deep.equal(COMMAND_TYPES.filter(type => types.includes(type)));
      expect(types[0]).to.equal('gameEndTurnRequest');
    });

    it('returns 1 only for complete attemptable leaves', () => {
      const game = buildGame(DEPLOY_SEED);
      const context = makeContext(game, 'p0', 'NORMAL', 203);
      const leaves = collectLeaves(domain, context);
      const deployLeaf = leaves.find(({command}) => command.type === 'gameDeployAnimalRequest');
      expect(deployLeaf).to.be.ok;

      deployLeaf.prefix.forEach((item, index) => {
        const prefix = deployLeaf.prefix.slice(0, index + 1);
        const complete = index === deployLeaf.prefix.length - 1;
        expect(domain.isAttemptable(prefix, context)).to.equal(complete ? 1 : 0);
      });

      expect(() => domain.resolveCommand([], context)).to.throw('INVALID_PREFIX');
      expect(() => domain.choices([{field_kind: 'TYPE', choice_kind: 'TYPE', type_id: 999}], context))
        .to.throw('INVALID_PREFIX');

      // Observation rows carry numeric schema enum ids; the domain accepts them.
      const root = domain.rootChoices(context);
      const deployType = root.find(item => item.type_name === 'gameDeployAnimalRequest');
      const numericRow = {
        field_kind: catalog.namespaces.choice_field_kinds.TYPE
        , choice_kind: catalog.namespaces.choice_kinds.TYPE
        , type_id: deployType.type_id
        , entity_kind: 0
        , entity_slot: -1
        , trait_slot: -1
        , scalar: 0
      };
      expect(domain.choices([numericRow], context))
        .to.deep.equal(domain.choices([deployType], context));
    });

    it('does not reveal opponent hand contents', () => {
      const left = buildGame(`
phase: deploy
deck: 10 camo
players:
  - continent: $A carn, $B
    hand: 2 camo, 1 communication
  - continent: $C, $D
    hand: 2 camo
`);
      const right = buildGame(`
phase: deploy
deck: 10 camo
players:
  - continent: $A carn, $B
    hand: 2 camo, 1 communication
  - continent: $C, $D
    hand: 2 mimicry
`);
      const leftContext = makeContext(left, 'p0', 'NORMAL', 210);
      const rightContext = makeContext(right, 'p0', 'NORMAL', 211);

      const compare = (prefix) => {
        const leftChoices = domain.choices(prefix, leftContext);
        const rightChoices = domain.choices(prefix, rightContext);
        expect(leftChoices).to.deep.equal(rightChoices);
        leftChoices.forEach((item) => {
          const next = [...prefix, item];
          if (domain.isAttemptable(next, leftContext) === 0) compare(next);
        });
      };
      compare([]);
    });
  });

  describe('special decisions', () => {
    it('DEFENSE offers active defenses and a decline', () => {
      const game = buildGame(`
phase: feeding
players:
  - continent: $A carn, $W wait
  - continent: $B run mimic, $C
`);
      const attackAnimal = game.locateAnimal('$A', 'p0');
      const targetAnimal = game.locateAnimal('$B', 'p1');
      const attackTrait = attackAnimal.traits.first();
      const question = new QuestionRecord({
        id: 'q_defense'
        , type: QuestionRecord.DEFENSE
        , userId: 'p1'
        , sourcePid: 'p0'
        , sourceAid: attackAnimal.id
        , traitId: attackTrait.id
        , targetPid: 'p1'
        , targetAid: targetAnimal.id
      });
      const withQuestion = game.set('question', question);
      const context = makeContext(withQuestion, 'p1', 'DEFENSE', 2, {question});
      const leaves = collectLeaves(domain, context);
      const traitTypes = leaves.map(({command}) => {
        const traitId = command.data.traitId;
        if (traitId === true) return true;
        const trait = targetAnimal.traits.get(traitId);
        return trait ? trait.type : traitId;
      });
      expect(traitTypes).to.include('TraitRunning');
      expect(traitTypes).to.include('TraitMimicry');
      expect(traitTypes).to.include(true);
    });

    it('INTELLECT offers target traits and a decline', () => {
      const game = buildGame(`
phase: feeding
players:
  - continent: $A carn int, $W wait
  - continent: $B run fat, $C
`);
      const attackAnimal = game.locateAnimal('$A', 'p0');
      const targetAnimal = game.locateAnimal('$B', 'p1');
      const attackTrait = attackAnimal.traits.first();
      const question = new QuestionRecord({
        id: 'q_intellect'
        , type: QuestionRecord.INTELLECT
        , userId: 'p0'
        , sourcePid: 'p0'
        , sourceAid: attackAnimal.id
        , traitId: attackTrait.id
        , targetPid: 'p1'
        , targetAid: targetAnimal.id
      });
      const withQuestion = game.set('question', question);
      const context = makeContext(withQuestion, 'p0', 'INTELLECT', 3, {question});
      const leaves = collectLeaves(domain, context);
      const targetTypes = leaves.map(({command}) => {
        const targetId = command.data.targetId;
        if (targetId === true) return true;
        const trait = targetAnimal.traits.get(targetId);
        return trait ? trait.type : targetId;
      });
      expect(targetTypes).to.include(true);
      expect(targetTypes).to.include('TraitRunning');
    });

    it('AMBUSH offers own pending ambushers and continue', () => {
      const game = buildGame(`
phase: ambush
players:
  - continent: $A carn amb, $W wait
  - continent: $B
`);
      const animal = game.locateAnimal('$A', 'p0');
      const ambush = {
        ambushers: new Map([[animal.id, null]])
      };
      const context = makeContext(game, 'p0', 'AMBUSH', 4, {ambush});
      const leaves = collectLeaves(domain, context);
      expect(leaves.some(({command}) => command.type === 'traitAmbushActivateRequest')).to.equal(true);
      expect(leaves.some(({command}) => command.type === 'traitAmbushContinueRequest')).to.equal(true);
    });

    it('REGENERATION offers flagged animals, hand cards and USE_DEFAULT', () => {
      const game = buildGame(`
phase: regeneration
players:
  - continent: $A carn
    hand: 2 camo
  - continent: $B
`);
      const animal = game.locateAnimal('$A', 'p0');
      const withFlag = game.updateIn(['players', 'p0', 'continent', animal.id]
        , current => current.set('flags', Map({REGENERATION: true})));
      const context = makeContext(withFlag, 'p0', 'REGENERATION', 5);
      const leaves = collectLeaves(domain, context);
      expect(leaves.filter(({command}) => command.type === 'gameDeployRegeneratedAnimalRequest')).to.have.length(2);
      expect(leaves.some(({command}) => command.special === 'USE_DEFAULT')).to.equal(true);
    });

    it('TRANSIENT asks one boolean per own Anglerfish and READY when empty', () => {
      const game = buildGame(`
phase: feeding
players:
  - continent: $A angler, $W wait
  - continent: $B
`);
      const context = makeContext(game, 'p0', 'TRANSIENT', 6);
      const leaves = collectLeaves(domain, context);
      expect(leaves).to.have.length(2);
      leaves.forEach(({command}) => {
        expect(command.special).to.equal('TRANSIENT');
        expect(command.vector).to.have.length(1);
        expect(command.vector[0].value).to.be.a('boolean');
      });

      const emptyContext = makeContext(buildGame(`
phase: feeding
players:
  - continent: $A carn, $W wait
  - continent: $B
`), 'p0', 'TRANSIENT', 7);
      const ready = collectLeaves(domain, emptyContext);
      expect(ready).to.have.length(1);
      expect(ready[0].command.special).to.equal('READY');
    });

    it('TERMINAL has no choices', () => {
      const game = buildGame(DEPLOY_SEED);
      const context = makeContext(game, 'p0', 'TERMINAL', 8);
      expect(domain.rootChoices(context)).to.deep.equal([]);
      expect(collectLeaves(domain, context)).to.deep.equal([]);
    });
  });
});
