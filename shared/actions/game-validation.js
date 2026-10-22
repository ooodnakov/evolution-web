/**
 * Pure guard extraction for gameplay requests.
 *
 * `validateGameRequest` mirrors, in the same order, the guards that the
 * authoritative request handlers run. It dispatches nothing, runs no
 * auto-sharing, draws no RNG and never mutates the game. Every failure is the
 * exact `ActionCheckError` (or `throwError` result) the handler would throw.
 *
 * The handlers are split in two groups around `server$autoFoodSharing`:
 *  - `before_sharing` — everything the handler checks before auto-sharing;
 *  - `after_sharing`  — everything checked after it (on the reread game).
 * `complete` runs before, then after, on the passed state (no effect
 * simulation), which is exactly what a handler without auto-sharing does.
 */
import {PHASE, QuestionRecord} from '../models/game/GameModel';
import ActionCheckError from '../models/ActionCheckError';
import {CTT_PARAMETER, TRAIT_ANIMAL_FLAG} from '../models/game/evolution/constants';
import * as ERR from '../errors/ERR';

import {
  checkGameDefined
  , checkGameHasUser
  , checkGamePhase
  , checkPlayerCanAct
  , checkPlayerHasCard
  , checkValidAnimalPosition
  , getOrThrowAnimalTrait
  , getOrThrowGameAnimal
  , getOrThrowGamePlant
  , getOrThrowPlayerAnimal
  , getOrThrowQuestion
  , throwError
} from './checks';
import {
  checkAnimalCanTakeShellFails
  , checkTraitActivation_Target
  , getErrorOfAnimalEatingFromGame
  , getErrorOfAnimalEatingFromPlant
  , getErrorOfAnimalTakingCover
  , getErrorOfEntityTraitActivation
  , getErrorOfPlantAttack
} from './trait.checks';

const STAGES = ['before_sharing', 'after_sharing', 'complete'];

/**
 * @param {object} game authoritative or projected game
 * @param {string} userId acting player
 * @param {{type: string}} request
 * @param {'before_sharing'|'after_sharing'|'complete'} stage
 * @param {object} [operands] resolved values from an earlier `before_sharing`
 *   call (used by handlers that reread the game after auto-sharing and keep
 *   the pre-sharing trait instance)
 * @returns {object} resolved operands
 */
export const validateGameRequest = (game, userId, request, stage = 'complete', operands = null) => {
  if (!STAGES.includes(stage)) {
    throw new Error('UNKNOWN_VALIDATION_STAGE');
  }
  if (!request || typeof request.type !== 'string') {
    throw new Error('UNKNOWN_REQUEST_TYPE');
  }
  const runBefore = stage === 'before_sharing' || stage === 'complete';
  const runAfter = stage === 'after_sharing' || stage === 'complete';
  const carried = operands || {};

  let resolved = {};

  if (runBefore) {
    resolved = beforeSharing(game, userId, request);
    if (stage === 'complete') {
      resolved = {...carried, ...resolved};
    }
  } else {
    resolved = {...carried};
  }

  if (runAfter) {
    resolved = {...resolved, ...afterSharing(game, userId, request, {...carried, ...resolved})};
  }

  return resolved;
};

const beforeSharing = (game, userId, request) => {
  switch (request.type) {
    case 'gameEndTurnRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkPlayerCanAct(game, userId);
      if (game.status.phase !== PHASE.FEEDING && game.status.phase !== PHASE.DEPLOY) {
        throw new ActionCheckError(ERR.GAME_PHASE, {
          gameId: game.id
          , phase: game.status.phase
          , targetPhase: `${PHASE.FEEDING}/${PHASE.DEPLOY}`
        });
      }
      return {};
    }

    case 'gameDeployAnimalRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.DEPLOY);
      checkPlayerCanAct(game, userId);
      checkValidAnimalPosition(game, userId, request.animalPosition);
      const card = checkPlayerHasCard(game, userId, request.cardId);
      return {card};
    }

    case 'gameDeployTraitRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.DEPLOY);
      checkPlayerCanAct(game, userId);

      const card = checkPlayerHasCard(game, userId, request.cardId);
      const traitData = card.getTraitDataModel(request.alternateTrait);
      if (!traitData) {
        throw new ActionCheckError(ERR.GAME_CARD_TRAIT_NOTFOUND, {
          gameId: game.id
          , userId
          , cardId: request.cardId
          , trait1: card.trait1
          , trait2: card.trait2
          , alternateTrait: request.alternateTrait
          , linkId: request.linkId
        });
      }

      let entity;
      let entityOwnerId;

      if (traitData.cardTargetType & CTT_PARAMETER.ANIMAL) {
        entity = game.locateAnimal(request.entityId);
        if (entity) {
          entityOwnerId = entity.ownerId;
        }
      } else if (traitData.cardTargetType & CTT_PARAMETER.PLANT) {
        entity = game.getPlant(request.entityId);
      }

      if (!entity) {
        throw new ActionCheckError(ERR.GAME_ENTITY_NOTFOUND, {
          gameId: game.id
          , entityId: request.entityId
        });
      }

      throwError(traitData.getErrorOfTraitPlacement(entity));
      throwError(traitData.getErrorOfTraitPlacement_User(userId, entityOwnerId));

      let linkedEntity;
      let linkedEntityOwnerId;
      if (traitData.linkTargetType) {
        if (traitData.linkTargetType & CTT_PARAMETER.PLANT) {
          linkedEntity = game.getPlant(request.linkId);
        } else if (traitData.linkTargetType & CTT_PARAMETER.ANIMAL) {
          linkedEntity = game.locateAnimal(request.linkId);
          if (linkedEntity) {
            linkedEntityOwnerId = linkedEntity.ownerId;
          }
        }

        if (!linkedEntity) {
          throw new ActionCheckError(ERR.GAME_LINK_NOTFOUND, {
            gameId: game.id
            , linkId: request.linkId
          });
        }
        if (entity === linkedEntity) {
          throw new ActionCheckError(ERR.GAME_LINK_SELF, {
            gameId: game.id
            , linkId: request.linkId
          });
        }
        throwError(traitData.getErrorOfTraitPlacement(linkedEntity));
        throwError(traitData.getErrorOfTraitPlacement_LinkUser(userId, linkedEntityOwnerId));
      }

      return {card, traitData, entity, linkedEntity};
    }

    case 'gameDeployRegeneratedAnimalRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.REGENERATION);
      const card = checkPlayerHasCard(game, userId, request.cardId);
      const animal = getOrThrowPlayerAnimal(game, userId, request.animalId);
      if (!animal.hasFlag(TRAIT_ANIMAL_FLAG.REGENERATION)) {
        throw new ActionCheckError(ERR.GAME_ANIMAL_FLAG, {
          gameId: game.id
          , userId
          , animalId: request.animalId
          , flag: TRAIT_ANIMAL_FLAG.REGENERATION
        });
      }
      return {card, animal};
    }

    case 'traitTakeFoodRequest': {
      // Auto-sharing runs first in the handler.
      return {};
    }

    case 'traitTakeShellRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.FEEDING);
      checkPlayerCanAct(game, userId);
      const animal = getOrThrowPlayerAnimal(game, userId, request.animalId);
      return {animal};
    }

    case 'traitTakeCoverRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.FEEDING);
      checkPlayerCanAct(game, userId);
      const animal = getOrThrowPlayerAnimal(game, userId, request.animalId);
      return {animal};
    }

    case 'traitActivateRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.FEEDING);
      const animal = getOrThrowPlayerAnimal(game, userId, request.sourceAid);
      const trait = getOrThrowAnimalTrait(game.id, animal, request.traitId);
      return {animal, trait};
    }

    case 'gamePlantAttackRequest': {
      // Auto-sharing runs first in the handler.
      return {};
    }

    case 'plantTraitActivateRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.FEEDING);
      const plant = getOrThrowGamePlant(game, request.plantId);
      const trait = plant.hasTrait(request.traitId);
      const targets = request.targets || [];
      throwError(getErrorOfEntityTraitActivation(game, userId, plant, trait, ...targets));
      if (trait && !trait.getDataModel().transient) checkPlayerCanAct(game, userId);
      return {plant, trait};
    }

    case 'traitAnswerRequest': {
      checkGameDefined(game);
      const question = getOrThrowQuestion(game, request.questionId);
      if (game.status.phase !== PHASE.FEEDING && game.status.phase !== PHASE.AMBUSH) {
        throw new ActionCheckError(ERR.GAME_PHASE, {
          gameId: game.id
          , phase: game.status.phase
          , targetPhase: `${PHASE.FEEDING}/${PHASE.AMBUSH}`
        });
      }
      const {type, sourcePid, targetPid} = question;
      switch (type) {
        case QuestionRecord.DEFENSE:
          if (userId !== targetPid) {
            throw new ActionCheckError(ERR.GAME_PLAYER_ACT_ON_QUESTION, {
              gameId: game.id
              , userId
              , targetPid
            });
          }
          break;
        case QuestionRecord.INTELLECT:
          if (userId !== sourcePid) {
            throw new ActionCheckError(ERR.GAME_PLAYER_ACT_ON_QUESTION, {
              gameId: game.id
              , userId
              , targetPid
            });
          }
          break;
      }
      return {question};
    }

    default:
      throw new Error('UNKNOWN_REQUEST_TYPE');
  }
};

const afterSharing = (game, userId, request, resolved) => {
  switch (request.type) {
    case 'traitTakeFoodRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.FEEDING);
      checkPlayerCanAct(game, userId);

      const animal = getOrThrowPlayerAnimal(game, userId, request.animalId);
      let plant = null;
      if (!game.isPlantarium()) {
        throwError(getErrorOfAnimalEatingFromGame(game, animal));
      } else {
        plant = getOrThrowGamePlant(game, request.plantId);
        throwError(getErrorOfAnimalEatingFromPlant(game, animal, plant));
      }
      return {animal, plant};
    }

    case 'traitTakeShellRequest': {
      const animal = resolved.animal || getOrThrowPlayerAnimal(game, userId, request.animalId);
      throwError(checkAnimalCanTakeShellFails(game, animal));
      const trait = game.getArea().shells.get(request.traitId);
      if (!trait) {
        throw new ActionCheckError(ERR.GAME_TRAIT_NOTFOUND, {
          gameId: game.id
          , traitId: request.traitId
        });
      }
      return {animal, trait};
    }

    case 'traitTakeCoverRequest': {
      const animal = resolved.animal || getOrThrowPlayerAnimal(game, userId, request.animalId);
      const plant = getOrThrowGamePlant(game, request.plantId);
      throwError(getErrorOfAnimalTakingCover(game, animal, plant));
      return {animal, plant};
    }

    case 'traitActivateRequest': {
      const animal = getOrThrowPlayerAnimal(game, userId, request.sourceAid);
      const trait = resolved.trait || getOrThrowAnimalTrait(game.id, animal, request.traitId);
      const targets = request.targets || [];
      throwError(trait.getErrorOfUse(game, animal, ...targets));
      const target = checkTraitActivation_Target(game, animal, trait, ...targets);
      if (!trait.getDataModel().transient) checkPlayerCanAct(game, userId);
      return {animal, trait, target};
    }

    case 'gamePlantAttackRequest': {
      checkGameDefined(game);
      checkGameHasUser(game, userId);
      checkGamePhase(game, PHASE.FEEDING);
      checkPlayerCanAct(game, userId);
      const animal = getOrThrowGameAnimal(game, request.targetId);
      const plant = getOrThrowGamePlant(game, request.plantId);
      throwError(getErrorOfPlantAttack(game, animal, plant, userId));
      return {animal, plant};
    }

    case 'plantTraitActivateRequest': {
      return {};
    }

    case 'gameEndTurnRequest':
    case 'gameDeployAnimalRequest':
    case 'gameDeployTraitRequest':
    case 'gameDeployRegeneratedAnimalRequest':
    case 'traitAnswerRequest':
      return {};

    default:
      throw new Error('UNKNOWN_REQUEST_TYPE');
  }
};

export default validateGameRequest;
