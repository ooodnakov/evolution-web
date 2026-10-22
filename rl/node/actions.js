/**
 * Prefix action domain for the headless RL environment.
 *
 * The policy never sends raw game ids. `createActionDomain` enumerates the
 * next observable choices for the acting player, level by level (type -> card
 * -> position -> entity -> target -> ...). Every emitted option has at least
 * one admissible completion; leaves are never truncated and distinct
 * positions/sides/link endpoints/shell ids are never deduplicated.
 *
 * Legality is computed on the player-information projection, never on the
 * authoritative game. The mask semantics are `public_preconditions`: a leaf is
 * admitted when the public preconditions hold. When auto food sharing is
 * pending, only the pre-sharing guards plus immutable shape/existence/type/
 * ownership constraints are used, because feeding/RNG outcomes cannot be
 * predicted without cheating.
 */
import {PHASE, QuestionRecord} from '~/shared/models/game/GameModel';
import {
  CTT_PARAMETER
  , HUNT_FLAG
  , TRAIT_ANIMAL_FLAG
  , TRAIT_TARGET_TYPE
} from '~/shared/models/game/evolution/constants';
import * as tt from '~/shared/models/game/evolution/traitTypes';
import * as ptt from '~/shared/models/game/evolution/plantarium/plantTraitTypes';
import {
  getActiveDefenses
} from '~/shared/models/game/evolution/traitsData/TraitCarnivorous';
import {
  checkIfTraitDisabledByIntellect
  , checkTraitActivation_Target
} from '~/shared/actions/trait.checks';
import {
  checkGameDefined
  , checkGameHasUser
  , checkGamePhase
  , checkPlayerCanAct
  , getOrThrowGameAnimal
  , getOrThrowGamePlant
  , getOrThrowPlayerAnimal
} from '~/shared/actions/checks';
import {validateGameRequest} from '~/shared/actions/game-validation';

import {entitiesOfGame} from './slots';
import {createCatalog, typeId as catalogTypeId} from './catalog';

export const COMMAND_TYPES = [
  'gameEndTurnRequest'
  , 'gameDeployAnimalRequest'
  , 'gameDeployTraitRequest'
  , 'traitTakeFoodRequest'
  , 'traitTakeShellRequest'
  , 'traitTakeCoverRequest'
  , 'traitActivateRequest'
  , 'gamePlantAttackRequest'
  , 'plantTraitActivateRequest'
];

export const DECISION_KINDS = [
  'NORMAL', 'DEFENSE', 'INTELLECT', 'AMBUSH', 'REGENERATION', 'TRANSIENT', 'TERMINAL'
];

export const ENTITY_KIND = {NONE: 0, ANIMAL: 1, PLANT: 2, SHELL: 3};

const SHARING_TYPES = new Set([
  'traitTakeFoodRequest'
  , 'traitTakeShellRequest'
  , 'traitTakeCoverRequest'
  , 'traitActivateRequest'
  , 'gamePlantAttackRequest'
]);

const num = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : -1);

const choice = (fields) => ({
  field_kind: 0
  , choice_kind: 0
  , type_id: 0
  , type_name: null
  , entity_kind: 0
  , entity_slot: -1
  , entity_id: null
  , trait_slot: -1
  , trait_id: null
  , scalar: 0
  , scalar_id: null
  , ...fields
});

const enumId = (catalog, namespace, value) => (
  typeof value === 'number' ? value : catalogTypeId(catalog, namespace, value)
);

/**
 * Stable key over fields that survive the observation projection.
 * Field/choice kinds may arrive as names (internal rows) or numeric schema
 * ids (rows returned by `observation.observe`), so both are normalized.
 */
export const choiceKey = (value, catalog = createCatalog()) => [
  enumId(catalog, 'choice_field_kinds', value && value.field_kind)
  , enumId(catalog, 'choice_kinds', value && value.choice_kind)
  , (value && value.type_id) | 0
  , (value && value.entity_kind) | 0
  , num(value && value.entity_slot)
  , num(value && value.trait_slot)
  , num(value && value.scalar)
].join('|');

const typeChoice = (type) => choice({
  field_kind: 'TYPE'
  , choice_kind: 'TYPE'
  , type_id: COMMAND_TYPES.indexOf(type) + 1
  , type_name: type
});

const handCardChoice = (card, index, resolveTypeId) => choice({
  field_kind: 'HAND_CARD'
  , choice_kind: 'ENTITY'
  , entity_slot: index
  , entity_id: card.id
  , type_id: resolveTypeId('card_types', card.type)
  , type_name: card.type
  , scalar: index
  , scalar_id: card.id
});

const positionChoice = (position) => choice({
  field_kind: 'POSITION'
  , choice_kind: 'SCALAR'
  , scalar: position
});

const booleanChoice = (value) => choice({
  field_kind: 'BOOLEAN'
  , choice_kind: 'SCALAR'
  , scalar: value ? 1 : 0
  , scalar_id: !!value
});

const animalChoice = (entry) => choice({
  field_kind: 'ANIMAL'
  , choice_kind: 'ENTITY'
  , entity_kind: ENTITY_KIND.ANIMAL
  , entity_slot: entry.slot
  , entity_id: entry.id
});

const plantChoice = (entry, resolveTypeId) => choice({
  field_kind: 'PLANT'
  , choice_kind: 'ENTITY'
  , entity_kind: ENTITY_KIND.PLANT
  , entity_slot: entry.slot
  , entity_id: entry.id
  , type_id: resolveTypeId('plant_types', entry.plant ? entry.plant.type : null)
  , type_name: entry.plant ? entry.plant.type : null
});

const shellChoice = (entry, resolveTypeId) => choice({
  field_kind: 'SHELL'
  , choice_kind: 'ENTITY'
  , entity_kind: ENTITY_KIND.SHELL
  , entity_slot: entry.slot
  , entity_id: entry.id
  , type_id: resolveTypeId('all_trait_types', entry.shell ? entry.shell.type : null)
  , type_name: entry.shell ? entry.shell.type : null
});

const entityChoice = (entry, kind) => choice({
  field_kind: 'ENTITY'
  , choice_kind: 'ENTITY'
  , entity_kind: kind
  , entity_slot: entry.slot
  , entity_id: entry.id
});

const traitChoice = (fieldKind, entityKind, entitySlot, trait, traitSlot, resolveTypeId) => choice({
  field_kind: fieldKind
  , choice_kind: 'TRAIT'
  , entity_kind: entityKind
  , entity_slot: entitySlot
  , trait_slot: traitSlot
  , trait_id: trait.id
  , type_id: resolveTypeId('all_trait_types', trait.type)
  , type_name: trait.type
});

const targetAnimalChoice = (entry) => choice({
  field_kind: 'TARGET'
  , choice_kind: 'ENTITY'
  , entity_kind: ENTITY_KIND.ANIMAL
  , entity_slot: entry.slot
  , entity_id: entry.id
});

const targetPlantChoice = (entry) => choice({
  field_kind: 'TARGET'
  , choice_kind: 'ENTITY'
  , entity_kind: ENTITY_KIND.PLANT
  , entity_slot: entry.slot
  , entity_id: entry.id
});

const declineChoice = () => choice({field_kind: 'DECLINE', choice_kind: 'DECLINE'});
const continueChoice = () => choice({
  field_kind: 'DECLINE'
  , choice_kind: 'DECLINE'
  , type_name: 'traitAmbushContinueRequest'
});
const useDefaultChoice = () => choice({field_kind: 'USE_DEFAULT', choice_kind: 'USE_DEFAULT'});
const readyChoice = () => choice({field_kind: 'READY', choice_kind: 'READY'});

const traitEntries = (entity) => entity.traits.toList()
  .map((trait, slot) => ({trait, slot}));

const linkedAnimalOf = (trait, sourceAnimalEntry, slots) => {
  const linkedId = sourceAnimalEntry.id === trait.hostAnimalId
    ? trait.linkAnimalId
    : trait.hostAnimalId;
  if (linkedId === null || linkedId === undefined) return null;
  return slots.animalById.get(linkedId) || null;
};

const targetCombinations = (targetType, sourceEntry, sourceKind, slots, resolveTypeId, trait) => {
  switch (targetType) {
    case TRAIT_TARGET_TYPE.NONE:
      return [{targets: [], choices: []}];
    case TRAIT_TARGET_TYPE.ANIMAL:
      return slots.animals.map(entry => ({
        targets: [entry.id]
        , choices: [targetAnimalChoice(entry)]
      }));
    case TRAIT_TARGET_TYPE.PLANT:
      return slots.plants.map(entry => ({
        targets: [entry.id]
        , choices: [targetPlantChoice(entry)]
      }));
    case TRAIT_TARGET_TYPE.TRAIT:
      return traitEntries(sourceEntry.entity).map(({trait: targetTrait, slot: targetSlot}) => ({
        targets: [targetTrait.id]
        , choices: [traitChoice('TARGET', sourceKind, sourceEntry.slot, targetTrait, targetSlot, resolveTypeId)]
      }));
    case TRAIT_TARGET_TYPE.TWO_TRAITS: {
      const linkedEntry = linkedAnimalOf(trait, sourceEntry, slots);
      if (!linkedEntry || !linkedEntry.animal) return [];
      const first = traitEntries(sourceEntry.entity);
      const second = traitEntries(linkedEntry.animal);
      const result = [];
      first.forEach(({trait: firstTrait, slot: firstSlot}) => {
        second.forEach(({trait: secondTrait, slot: secondSlot}) => {
          result.push({
            targets: [firstTrait.id, secondTrait.id]
            , choices: [
              traitChoice('TARGET', sourceKind, sourceEntry.slot, firstTrait, firstSlot, resolveTypeId)
              , traitChoice('TARGET', ENTITY_KIND.ANIMAL, linkedEntry.slot, secondTrait, secondSlot, resolveTypeId)
            ]
          });
        });
      });
      return result;
    }
    default:
      return [];
  }
};

const hasAvailableSharingTrait = (game, playerId) => {
  const player = game && game.getPlayer ? game.getPlayer(playerId) : null;
  if (!player) return false;
  let available = false;
  player.someAnimal((animal) => animal.getTraits().some((trait) => {
    if (trait.type !== tt.TraitCommunication && trait.type !== tt.TraitCooperation) return false;
    let error;
    try {
      error = trait.getErrorOfUse(game, animal);
    } catch (e) {
      return false;
    }
    if (error) return false;
    available = true;
    return true;
  }));
  return available;
};

const sourceTraitOf = (game, playerId, sourceAid, traitId) => {
  const animal = game && game.locateAnimal(sourceAid, playerId);
  if (!animal) return null;
  return animal.traits.get(traitId)
    || animal.traits.find(trait => trait.type === traitId)
    || null;
};

/** Does this request run `server$autoFoodSharing` with a usable trait? */
export const preparationPending = (request, context) => {
  if (!request || !SHARING_TYPES.has(request.type)) return false;
  const game = context.projection || context.game;
  if (!game) return false;
  if (request.type === 'traitActivateRequest') {
    const trait = sourceTraitOf(game, context.playerId, request.sourceAid, request.traitId);
    if (!trait) return false;
    if (trait.type === tt.TraitCommunication || trait.type === tt.TraitCooperation) return false;
  }
  return hasAvailableSharingTrait(game, context.playerId);
};

/**
 * Immutable shape/existence/ownership checks for request types whose real
 * before-sharing group is empty (auto-sharing is the first handler step).
 */
const validatePendingStructure = (request, context) => {
  const game = context.projection || context.game;
  const playerId = context.playerId;
  switch (request.type) {
    case 'traitTakeFoodRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, playerId);
      checkGamePhase(game, PHASE.FEEDING);
      checkPlayerCanAct(game, playerId);
      getOrThrowPlayerAnimal(game, playerId, request.animalId);
      if (game.isPlantarium()) {
        getOrThrowGamePlant(game, request.plantId);
      }
      return;
    }
    case 'gamePlantAttackRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, playerId);
      checkGamePhase(game, PHASE.FEEDING);
      checkPlayerCanAct(game, playerId);
      getOrThrowGameAnimal(game, request.targetId);
      getOrThrowGamePlant(game, request.plantId);
      return;
    }
    default:
      return;
  }
};

const leafAdmissible = (request, context) => {
  const game = context.projection || context.game;
  try {
    if (preparationPending(request, context)) {
      validateGameRequest(game, context.playerId, request, 'before_sharing');
      validatePendingStructure(request, context);
    } else {
      validateGameRequest(game, context.playerId, request, 'complete');
    }
    return true;
  } catch (error) {
    if (error && error.userLevelError) return false;
    throw error;
  }
};

const requestCommand = (request) => {
  const {type, gameId, ...data} = request;
  return {type, data};
};

const normalLeaf = (type, choiceFields, request, leaves, context) => {
  if (!leafAdmissible(request, context)) return;
  leaves.push({
    choices: [typeChoice(type), ...choiceFields]
    , request
    , command: requestCommand(request)
  });
};

const buildEndTurnLeaves = (leaves, context) => {
  const request = {type: 'gameEndTurnRequest'};
  normalLeaf('gameEndTurnRequest', [], request, leaves, context);
};

const buildDeployAnimalLeaves = (leaves, context) => {
  const game = context.projection;
  const player = game.getPlayer(context.playerId);
  const resolveTypeId = context.resolveTypeId;
  player.hand.toList().forEach((card, index) => {
    const cardChoice = handCardChoice(card, index, resolveTypeId);
    for (let position = 0; position <= player.continent.size; position += 1) {
      const request = {
        type: 'gameDeployAnimalRequest'
        , cardId: card.id
        , animalPosition: position
      };
      normalLeaf('gameDeployAnimalRequest', [cardChoice, positionChoice(position)], request, leaves, context);
    }
  });
};

const buildDeployTraitLeaves = (leaves, context) => {
  const game = context.projection;
  const player = game.getPlayer(context.playerId);
  const slots = context.slots;
  const resolveTypeId = context.resolveTypeId;

  player.hand.toList().forEach((card, index) => {
    const sides = [];
    if (card.trait1) sides.push(false);
    if (card.trait2) sides.push(true);

    sides.forEach((alternateTrait) => {
      const traitData = card.getTraitDataModel(alternateTrait);
      if (!traitData) return;

      let entityEntries = [];
      let entityKind = ENTITY_KIND.NONE;
      if (traitData.cardTargetType & CTT_PARAMETER.ANIMAL) {
        entityEntries = slots.animals;
        entityKind = ENTITY_KIND.ANIMAL;
      } else if (traitData.cardTargetType & CTT_PARAMETER.PLANT) {
        entityEntries = slots.plants;
        entityKind = ENTITY_KIND.PLANT;
      }
      if (entityEntries.length === 0) return;

      const cardChoice = handCardChoice(card, index, resolveTypeId);
      const sideChoice = booleanChoice(alternateTrait);

      entityEntries.forEach((entityEntry) => {
        const placeChoice = entityChoice(entityEntry, entityKind);
        if (!traitData.linkTargetType) {
          const request = {
            type: 'gameDeployTraitRequest'
            , cardId: card.id
            , entityId: entityEntry.id
            , alternateTrait
          };
          normalLeaf('gameDeployTraitRequest'
            , [cardChoice, sideChoice, placeChoice]
            , request
            , leaves
            , context);
          return;
        }

        let linkEntries = [];
        if (traitData.linkTargetType & CTT_PARAMETER.PLANT) {
          linkEntries = slots.plants;
        } else if (traitData.linkTargetType & CTT_PARAMETER.ANIMAL) {
          linkEntries = slots.animals;
        }

        linkEntries.forEach((linkEntry) => {
          const request = {
            type: 'gameDeployTraitRequest'
            , cardId: card.id
            , entityId: entityEntry.id
            , alternateTrait
            , linkId: linkEntry.id
          };
          normalLeaf('gameDeployTraitRequest'
            , [cardChoice, sideChoice, placeChoice, entityChoice(linkEntry, entityKind)]
            , request
            , leaves
            , context);
        });
      });
    });
  });
};

const buildTakeFoodLeaves = (leaves, context) => {
  const game = context.projection;
  const slots = context.slots;
  const ownAnimals = slots.animals.filter(entry => entry.ownerId === context.playerId);

  ownAnimals.forEach((animalEntry) => {
    if (!game.isPlantarium()) {
      const request = {
        type: 'traitTakeFoodRequest'
        , animalId: animalEntry.id
      };
      normalLeaf('traitTakeFoodRequest', [animalChoice(animalEntry)], request, leaves, context);
      return;
    }
    slots.plants.forEach((plantEntry) => {
      const request = {
        type: 'traitTakeFoodRequest'
        , animalId: animalEntry.id
        , plantId: plantEntry.id
      };
      normalLeaf('traitTakeFoodRequest'
        , [animalChoice(animalEntry), plantChoice(plantEntry, context.resolveTypeId)]
        , request
        , leaves
        , context);
    });
  });
};

const buildTakeShellLeaves = (leaves, context) => {
  const slots = context.slots;
  const ownAnimals = slots.animals.filter(entry => entry.ownerId === context.playerId);
  ownAnimals.forEach((animalEntry) => {
    slots.shells.forEach((shellEntry) => {
      const request = {
        type: 'traitTakeShellRequest'
        , animalId: animalEntry.id
        , traitId: shellEntry.id
      };
      normalLeaf('traitTakeShellRequest'
        , [animalChoice(animalEntry), shellChoice(shellEntry, context.resolveTypeId)]
        , request
        , leaves
        , context);
    });
  });
};

const buildTakeCoverLeaves = (leaves, context) => {
  const slots = context.slots;
  const ownAnimals = slots.animals.filter(entry => entry.ownerId === context.playerId);
  ownAnimals.forEach((animalEntry) => {
    slots.plants.forEach((plantEntry) => {
      const request = {
        type: 'traitTakeCoverRequest'
        , animalId: animalEntry.id
        , plantId: plantEntry.id
      };
      normalLeaf('traitTakeCoverRequest'
        , [animalChoice(animalEntry), plantChoice(plantEntry, context.resolveTypeId)]
        , request
        , leaves
        , context);
    });
  });
};

const buildActivateLeaves = (leaves, context) => {
  const slots = context.slots;
  const resolveTypeId = context.resolveTypeId;
  const ownAnimals = slots.animals.filter(entry => entry.ownerId === context.playerId);

  ownAnimals.forEach((animalEntry) => {
    traitEntries(animalEntry.animal).forEach(({trait, slot}) => {
      const traitData = trait.getDataModel();
      if (!traitData) return;
      // Same filter as the web option search: defense-only and transient
      // traits are resolved through questions / the transient window, and
      // non-controllable traits are not player actions at all.
      if (traitData.defense || traitData.transient || !traitData.playerControllable) return;
      if (trait.getErrorOfUse(context.projection, animalEntry.animal)) return;
      const combinations = targetCombinations(
        traitData.targetType
        , {...animalEntry, entity: animalEntry.animal}
        , ENTITY_KIND.ANIMAL
        , slots
        , resolveTypeId
        , trait
      );
      combinations.forEach(({targets, choices: targetChoices}) => {
        const request = {
          type: 'traitActivateRequest'
          , sourceAid: animalEntry.id
          , traitId: trait.id
          , targets
        };
        normalLeaf('traitActivateRequest'
          , [
            animalChoice(animalEntry)
            , traitChoice('TRAIT', ENTITY_KIND.ANIMAL, animalEntry.slot, trait, slot, resolveTypeId)
            , ...targetChoices
          ]
          , request
          , leaves
          , context);
      });
    });
  });
};

const buildPlantAttackLeaves = (leaves, context) => {
  const slots = context.slots;
  slots.plants
    .filter(entry => entry.plant.hasTrait(ptt.PlantTraitHiddenCarnivorous))
    .forEach((plantEntry) => {
      slots.animals.forEach((animalEntry) => {
        const request = {
          type: 'gamePlantAttackRequest'
          , plantId: plantEntry.id
          , targetId: animalEntry.id
        };
        normalLeaf('gamePlantAttackRequest'
          , [plantChoice(plantEntry, context.resolveTypeId), targetAnimalChoice(animalEntry)]
          , request
          , leaves
          , context);
      });
    });
};

const buildPlantTraitLeaves = (leaves, context) => {
  const slots = context.slots;
  const resolveTypeId = context.resolveTypeId;

  slots.plants.forEach((plantEntry) => {
    const plantTraitEntries = traitEntries(plantEntry.plant);
    plantTraitEntries.forEach(({trait, slot}) => {
      const traitData = trait.getDataModel();
      if (!traitData || !traitData.playerControllable) return;
      if (traitData.defense || traitData.transient) return;
      const combinations = targetCombinations(
        traitData.targetType
        , {...plantEntry, entity: plantEntry.plant}
        , ENTITY_KIND.PLANT
        , slots
        , resolveTypeId
        , trait
      );
      combinations.forEach(({targets, choices: targetChoices}) => {
        const request = {
          type: 'plantTraitActivateRequest'
          , plantId: plantEntry.id
          , traitId: trait.id
          , targets
        };
        normalLeaf('plantTraitActivateRequest'
          , [
            plantChoice(plantEntry, context.resolveTypeId)
            , traitChoice('TRAIT', ENTITY_KIND.PLANT, plantEntry.slot, trait, slot, resolveTypeId)
            , ...targetChoices
          ]
          , request
          , leaves
          , context);
      });
    });
  });
};

const NORMAL_BUILDERS = {
  gameEndTurnRequest: buildEndTurnLeaves
  , gameDeployAnimalRequest: buildDeployAnimalLeaves
  , gameDeployTraitRequest: buildDeployTraitLeaves
  , traitTakeFoodRequest: buildTakeFoodLeaves
  , traitTakeShellRequest: buildTakeShellLeaves
  , traitTakeCoverRequest: buildTakeCoverLeaves
  , traitActivateRequest: buildActivateLeaves
  , gamePlantAttackRequest: buildPlantAttackLeaves
  , plantTraitActivateRequest: buildPlantTraitLeaves
};

const buildNormalLeaves = (context) => {
  const leaves = [];
  const game = context.projection || context.game;
  if (!game || !game.getPlayer || !game.getPlayer(context.playerId)) return leaves;
  COMMAND_TYPES.forEach((type) => {
    NORMAL_BUILDERS[type](leaves, context);
  });
  return leaves;
};

const defenseLeafAdmissible = (game, defenseAnimal, trait, targetId, attackAnimal, attackTrait) => {
  try {
    if (trait.getErrorOfUse(game, defenseAnimal, attackAnimal, attackTrait)) return false;
    if (checkIfTraitDisabledByIntellect(attackAnimal, trait)) return false;
    if (targetId === undefined) {
      return true;
    }
    checkTraitActivation_Target(game, defenseAnimal, trait, targetId, attackAnimal, attackTrait);
    return true;
  } catch (error) {
    if (error && error.userLevelError) return false;
    throw error;
  }
};

const buildDefenseLeaves = (context) => {
  const leaves = [];
  const game = context.projection || context.game;
  const question = context.question || (game && game.question);
  if (!question || question.type !== QuestionRecord.DEFENSE) return leaves;
  if (context.playerId !== question.targetPid) return leaves;

  const defenseAnimal = game.locateAnimal(question.targetAid, question.targetPid);
  const attackAnimal = game.locateAnimal(question.sourceAid, question.sourcePid);
  const attackTrait = game.locateTrait(question.traitId, question.sourceAid, question.sourcePid);
  if (!defenseAnimal || !attackAnimal || !attackTrait) return leaves;

  const slots = context.slots;
  const defenseEntry = slots.animalById.get(defenseAnimal.id);
  if (!defenseEntry) return leaves;

  const skipOptionalDefence = !!(game.hunts && game.hunts.size > 0
    && game.hunts.first().flags.has(HUNT_FLAG.OPTIONAL_DEFENCE_OFF));
  const defenses = getActiveDefenses(game, attackAnimal, attackTrait, defenseAnimal, skipOptionalDefence);

  defenses.forEach((trait) => {
    const traitData = trait.getDataModel();
    if (!traitData || !traitData.defense) return;
    if (checkIfTraitDisabledByIntellect(attackAnimal, trait)) return;

    let targetOptions = [];
    switch (traitData.targetType) {
      case TRAIT_TARGET_TYPE.NONE:
        targetOptions = [{targetId: undefined, choice: null}];
        break;
      case TRAIT_TARGET_TYPE.ANIMAL: {
        const list = traitData.getTargets
          ? traitData.getTargets(game, defenseAnimal, trait, attackAnimal, attackTrait).toList()
          : slots.animals.map(entry => entry.animal);
        targetOptions = list.map(target => ({
          targetId: target.id
          , choice: targetAnimalChoice(slots.animalById.get(target.id) || {
            slot: -1, id: target.id, animal: target
          })
        }));
        break;
      }
      case TRAIT_TARGET_TYPE.TRAIT: {
        const list = traitData.getTargets
          ? traitData.getTargets(game, defenseAnimal, trait, attackAnimal, attackTrait).toList()
          : defenseAnimal.traits.toList();
        targetOptions = list.map(targetTrait => {
          const slot = defenseAnimal.traits.toList().findIndex(item => item.id === targetTrait.id);
          return {
            targetId: targetTrait.id
            , choice: traitChoice('TARGET', ENTITY_KIND.ANIMAL, defenseEntry.slot, targetTrait, slot, context.resolveTypeId)
          };
        });
        break;
      }
      case TRAIT_TARGET_TYPE.PLANT:
        targetOptions = slots.plants.map(entry => ({
          targetId: entry.id
          , choice: targetPlantChoice(entry)
        }));
        break;
      default:
        return;
    }

    const traitSlot = defenseEntry.animal.traits.toList().findIndex(item => item.id === trait.id);
    targetOptions.forEach(({targetId, choice: targetChoice}) => {
      if (!defenseLeafAdmissible(game, defenseAnimal, trait, targetId, attackAnimal, attackTrait)) return;
      leaves.push({
        choices: [
          traitChoice('TRAIT', ENTITY_KIND.ANIMAL, defenseEntry.slot, trait, traitSlot, context.resolveTypeId)
          , ...(targetChoice ? [targetChoice] : [])
        ]
        , request: {
          type: 'traitAnswerRequest'
          , questionId: question.id
          , traitId: trait.id
          , targetId
        }
        , command: {
          type: 'traitAnswerRequest'
          , data: {questionId: question.id, traitId: trait.id, targetId}
        }
      });
    });
  });

  leaves.push({
    choices: [declineChoice()]
    , request: {
      type: 'traitAnswerRequest'
      , questionId: question.id
      , traitId: true
    }
    , command: {
      type: 'traitAnswerRequest'
      , data: {questionId: question.id, traitId: true}
    }
  });

  return leaves;
};

const buildIntellectLeaves = (context) => {
  const leaves = [];
  const game = context.projection || context.game;
  const question = context.question || (game && game.question);
  if (!question || question.type !== QuestionRecord.INTELLECT) return leaves;
  if (context.playerId !== question.sourcePid) return leaves;

  const targetAnimal = game.locateAnimal(question.targetAid, question.targetPid);
  const slots = context.slots;
  const targetEntry = targetAnimal ? slots.animalById.get(targetAnimal.id) : null;

  if (targetAnimal && targetEntry) {
    traitEntries(targetAnimal).forEach(({trait, slot}) => {
      leaves.push({
        choices: [
          traitChoice('TARGET', ENTITY_KIND.ANIMAL, targetEntry.slot, trait, slot, context.resolveTypeId)
        ]
        , request: {
          type: 'traitAnswerRequest'
          , questionId: question.id
          , traitId: tt.TraitIntellect
          , targetId: trait.id
        }
        , command: {
          type: 'traitAnswerRequest'
          , data: {questionId: question.id, traitId: tt.TraitIntellect, targetId: trait.id}
        }
      });
    });
  }

  leaves.push({
    choices: [declineChoice()]
    , request: {
      type: 'traitAnswerRequest'
      , questionId: question.id
      , traitId: tt.TraitIntellect
      , targetId: true
    }
    , command: {
      type: 'traitAnswerRequest'
      , data: {questionId: question.id, traitId: tt.TraitIntellect, targetId: true}
    }
  });

  return leaves;
};

const buildAmbushLeaves = (context) => {
  const leaves = [];
  const game = context.projection || context.game;
  const ambush = context.ambush || (game && game.ambush);
  if (!ambush) return leaves;

  const slots = context.slots;
  let hasPending = false;
  slots.animals
    .filter(entry => entry.ownerId === context.playerId)
    .forEach((entry) => {
      if (!ambush.ambushers.has(entry.id)) return;
      if (ambush.ambushers.get(entry.id) !== null) return;
      hasPending = true;
      [true, false].forEach((on) => {
        leaves.push({
          choices: [animalChoice(entry), booleanChoice(on)]
          , request: {type: 'traitAmbushActivateRequest', animalId: entry.id, on}
          , command: {type: 'traitAmbushActivateRequest', data: {animalId: entry.id, on}}
        });
      });
    });

  if (hasPending) {
    leaves.push({
      choices: [continueChoice()]
      , request: {type: 'traitAmbushContinueRequest'}
      , command: {type: 'traitAmbushContinueRequest', data: {}}
    });
  }
  return leaves;
};

const buildRegenerationLeaves = (context) => {
  const leaves = [];
  const game = context.projection || context.game;
  const player = game && game.getPlayer ? game.getPlayer(context.playerId) : null;
  if (!player) return leaves;
  const slots = context.slots;

  const ownAnimals = slots.animals
    .filter(entry => entry.ownerId === context.playerId)
    .filter(entry => entry.animal.hasFlag(TRAIT_ANIMAL_FLAG.REGENERATION));

  ownAnimals.forEach((animalEntry) => {
    player.hand.toList().forEach((card, index) => {
      const request = {
        type: 'gameDeployRegeneratedAnimalRequest'
        , cardId: card.id
        , animalId: animalEntry.id
      };
      try {
        validateGameRequest(game, context.playerId, request, 'complete');
      } catch (error) {
        if (error && error.userLevelError) return;
        throw error;
      }
      leaves.push({
        choices: [
          animalChoice(animalEntry)
          , handCardChoice(card, index, context.resolveTypeId)
        ]
        , request
        , command: requestCommand(request)
      });
    });
  });

  leaves.push({
    choices: [useDefaultChoice()]
    , request: {special: 'USE_DEFAULT'}
    , command: {special: 'USE_DEFAULT'}
  });
  return leaves;
};

const transientItems = (context) => {
  const game = context.projection || context.game;
  const slots = context.slots;
  const configured = context.transient;
  let items = null;
  const alreadySet = new Set();

  if (Array.isArray(configured)) {
    items = configured;
  } else if (configured && Array.isArray(configured.pending)) {
    items = configured.pending;
  }

  if (configured && configured.values) {
    const values = configured.values;
    if (typeof values.forEach === 'function' && typeof values.get === 'function') {
      values.forEach((value, key) => alreadySet.add(key));
    } else if (typeof values === 'object') {
      Object.keys(values).forEach(key => alreadySet.add(key));
    }
  }

  if (!items) {
    items = [];
    slots.animals
      .filter(entry => entry.ownerId === context.playerId)
      .forEach((entry) => {
        traitEntries(entry.animal).forEach(({trait}) => {
          if (trait.type !== tt.TraitAnglerfish) return;
          items.push({animalId: entry.id, traitId: trait.id});
        });
      });
  }

  return items
    .map(item => ({
      animalId: item.animalId || (item.animal && item.animal.id)
      , traitId: item.traitId || (item.trait && item.trait.id)
    }))
    .filter(item => item.animalId && item.traitId)
    .filter(item => !alreadySet.has(item.animalId) && !alreadySet.has(item.traitId));
};

const buildTransientLeaves = (context) => {
  const items = transientItems(context);
  if (items.length === 0) {
    return [{
      choices: [readyChoice()]
      , request: {special: 'READY'}
      , command: {special: 'READY'}
    }];
  }

  let prefixes = [[]];
  items.forEach(() => {
    const next = [];
    prefixes.forEach((prefix) => {
      [true, false].forEach((value) => next.push([...prefix, booleanChoice(value)]));
    });
    prefixes = next;
  });

  return prefixes.map((prefix) => {
    const values = prefix.map(item => item.scalar === 1);
    return {
      choices: prefix
      , request: {special: 'TRANSIENT', vector: items.map((item, index) => ({
        animalId: item.animalId
        , traitId: item.traitId
        , value: values[index]
      }))}
      , command: {special: 'TRANSIENT', vector: items.map((item, index) => ({
        animalId: item.animalId
        , traitId: item.traitId
        , value: values[index]
      }))}
    };
  });
};

const buildLeaves = (context) => {
  const kind = String(context.decisionKind || 'NORMAL').toUpperCase();
  switch (kind) {
    case 'TERMINAL':
      return [];
    case 'DEFENSE':
      return buildDefenseLeaves(context);
    case 'INTELLECT':
      return buildIntellectLeaves(context);
    case 'AMBUSH':
      return buildAmbushLeaves(context);
    case 'REGENERATION':
      return buildRegenerationLeaves(context);
    case 'TRANSIENT':
      return buildTransientLeaves(context);
    case 'NORMAL':
      return buildNormalLeaves(context);
    default:
      throw new Error('INVALID_DECISION_KIND');
  }
};

const normalizePrefix = (prefix) => (Array.isArray(prefix) ? prefix : []);

const matches = (keyOf, leafChoices, prefix) => {
  if (prefix.length > leafChoices.length) return false;
  for (let index = 0; index < prefix.length; index += 1) {
    if (keyOf(leafChoices[index]) !== keyOf(prefix[index])) return false;
  }
  return true;
};

const typeIdResolver = (catalog) => (namespace, name) => catalogTypeId(catalog, namespace, name);

export const createActionDomain = (options = {}) => {
  const catalog = (options && options.catalog) || createCatalog();
  const resolveTypeId = typeIdResolver(catalog);

  let cachedRevision;
  let cache = new Map();
  const keyOf = (value) => choiceKey(value, catalog);

  const leavesFor = (rawContext) => {
    const context = {
      ...rawContext
      , projection: rawContext.projection || (rawContext.game && rawContext.game.toOthers(rawContext.playerId))
      , slots: rawContext.slots || entitiesOfGame(rawContext.projection || rawContext.game)
      , resolveTypeId
    };
    const revision = context.revision;
    let key = null;
    if (revision !== undefined && revision !== null) {
      key = `${context.decisionKind}|${context.playerId}|${revision}`;
      if (cachedRevision !== revision) {
        cache.clear();
        cachedRevision = revision;
      }
      if (cache.has(key)) return cache.get(key);
    }
    const leaves = buildLeaves(context);
    if (key) cache.set(key, leaves);
    return leaves;
  };

  const rootChoices = (rawContext) => choices([], rawContext);

  function choices(prefix, rawContext) {
    const normalized = normalizePrefix(prefix);
    const leaves = leavesFor(rawContext);
    const matching = leaves.filter(leaf => matches(keyOf, leaf.choices, normalized));

    if (matching.length === 0) {
      if (normalized.length === 0) return [];
      throw new Error('INVALID_PREFIX');
    }

    const level = normalized.length;
    if (matching.every(leaf => leaf.choices.length <= level)) {
      return [];
    }

    const seen = new Set();
    const result = [];
    matching.forEach((leaf) => {
      if (leaf.choices.length <= level) return;
      const next = leaf.choices[level];
      const key = keyOf(next);
      if (seen.has(key)) return;
      seen.add(key);
      result.push(next);
    });
    return result;
  }

  const isComplete = (prefix, rawContext) => {
    const normalized = normalizePrefix(prefix);
    return leavesFor(rawContext).some(leaf => (
      leaf.choices.length === normalized.length && matches(keyOf, leaf.choices, normalized)
    ));
  };

  const resolveCommand = (prefix, rawContext) => {
    const normalized = normalizePrefix(prefix);
    const leaf = leavesFor(rawContext).find(candidate => (
      candidate.choices.length === normalized.length && matches(keyOf, candidate.choices, normalized)
    ));
    if (!leaf) throw new Error('INVALID_PREFIX');
    return leaf.command;
  };

  const isAttemptable = (prefix, rawContext) => (isComplete(prefix, rawContext) ? 1 : 0);

  return {
    rootChoices
    , choices
    , isComplete
    , resolveCommand
    , isAttemptable
  };
};

export default createActionDomain;
