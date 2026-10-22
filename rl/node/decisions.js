import {PHASE, QuestionRecord} from '~/shared/models/game/GameModel';
import {TRAIT_ANIMAL_FLAG} from '~/shared/models/game/evolution/constants';
import * as tt from '~/shared/models/game/evolution/traitTypes';

import {entitiesOfGame} from './slots';

/**
 * Pure decision discovery. No dispatch, no RNG, no mutation: given the current
 * authoritative game, return the next thing the environment has to ask.
 *
 * Return shape:
 *   {terminal: true}
 *   {stall: reason}
 *   {kind, actor, question?, sourceAid?, targetAid?, attackTraitType?}
 *   {regenerationEnd: true}
 */

const stall = (reason) => ({stall: reason});

export const questionKind = (question) => {
  if (!question) return null;
  if (question.type === QuestionRecord.DEFENSE) return 'DEFENSE';
  if (question.type === QuestionRecord.INTELLECT) return 'INTELLECT';
  return null;
};

const traitTypeOf = (game, animalId, ownerId, traitId) => {
  const animal = game.locateAnimal(animalId, ownerId);
  if (!animal) return -1;
  const trait = animal.traits.get(traitId) || animal.traits.find(candidate => candidate.type === traitId);
  return trait ? trait.type : -1;
};

const findQuestion = (game) => {
  const question = game.question;
  if (!question) return null;
  const kind = questionKind(question);
  if (!kind) return {stall: `unknown question type ${question.type}`};

  if (kind === 'DEFENSE') {
    return {
      kind
      , actor: question.userId
      , question
      , sourceAid: question.sourceAid
      , sourcePid: question.sourcePid
      , targetAid: question.targetAid
      , targetPid: question.targetPid
      , attackTraitType: traitTypeOf(game, question.sourceAid, question.sourcePid, question.traitId)
    };
  }

  return {
    kind
    , actor: question.userId
    , question
    , sourceAid: question.sourceAid
    , sourcePid: question.sourcePid
    , targetAid: question.targetAid
    , targetPid: question.targetPid
    , attackTraitType: tt.TraitIntellect
  };
};

const findAmbush = (game) => {
  const ambush = game.ambush;
  if (!ambush) return stall('ambush phase without ambush record');
  const pending = ambush.ambushers.filter(value => value === null);
  if (pending.size === 0) {
    const allAnswered = ambush.ambushers.every(value => value !== null);
    if (allAnswered && ambush.ambushers.size > 0) {
      return {ambushPrepareEnd: true};
    }
    return stall('ambush phase without pending ambushers');
  }

  const slots = entitiesOfGame(game);
  for (const agentSeat of game.players.valueSeq()) {
    for (const entry of slots.animals) {
      if (entry.ownerSeat !== agentSeat.index) continue;
      if (!pending.has(entry.id)) continue;
      const targetRecord = ambush.targets.first();
      return {
        kind: 'AMBUSH'
        , actor: entry.ownerId
        , ambush
        , sourceAid: entry.id
        , targetAid: targetRecord ? targetRecord.targetAid : null
      };
    }
  }
  return stall('ambusher owner not found');
};

const findRegeneration = (game, regenerationDone) => {
  const slots = entitiesOfGame(game);
  for (const player of game.players.valueSeq()) {
    if (regenerationDone && regenerationDone.has(player.id)) continue;
    if (player.hand.size <= 1) continue;
    const flagged = slots.animals.find(entry => (
      entry.ownerId === player.id && entry.animal.hasFlag(TRAIT_ANIMAL_FLAG.REGENERATION)
    ));
    if (flagged) {
      return {kind: 'REGENERATION', actor: player.id, sourceAid: flagged.id};
    }
  }
  return {regenerationEnd: true};
};

/** Anglerfish traits the player may still toggle in this window. */
export const availableTransientItems = (game, playerId) => {
  const slots = entitiesOfGame(game);
  const items = [];
  slots.animals
    .filter(entry => entry.ownerId === playerId)
    .forEach((entry) => {
      entry.animal.traits.forEach((trait) => {
        if (trait.type !== tt.TraitAnglerfish) return;
        if (trait.disabled) return;
        const error = trait.getErrorOfUse(game, entry.animal);
        if (error) return;
        items.push({animalId: entry.id, traitId: trait.id, slot: entry.slot});
      });
    });
  return items;
};

export const transientItemsBySeat = (game, agents) => {
  const result = {};
  agents.forEach((agentId) => {
    result[agentId] = availableTransientItems(game, agentId);
  });
  return result;
};

export const computePrimary = ({game, regenerationDone}) => {
  if (!game) return stall('no game');
  if (game.status.phase === PHASE.FINAL) return {terminal: true};

  const question = findQuestion(game);
  if (question) return question;

  if (game.status.phase === PHASE.AMBUSH) return findAmbush(game);
  if (game.status.phase === PHASE.REGENERATION) return findRegeneration(game, regenerationDone);

  if (game.status.phase === PHASE.DEPLOY || game.status.phase === PHASE.FEEDING) {
    const currentPlayer = game.status.currentPlayer;
    if (!currentPlayer || !game.getPlayer(currentPlayer)) {
      return stall(`no current player in ${game.status.phase}`);
    }
    return {kind: 'NORMAL', actor: currentPlayer};
  }

  return stall(`phase ${game.status.phase} requires no policy action but did not transition`);
};

export const primaryKey = (primary) => {
  if (!primary) return 'none';
  if (primary.terminal) return 'terminal';
  if (primary.regenerationEnd) return 'regen-end';
  if (primary.stall) return `stall:${primary.stall}`;
  return [
    primary.kind
    , primary.actor
    , primary.question ? primary.question.id : ''
    , primary.sourceAid || ''
    , primary.targetAid || ''
  ].join('|');
};

export default computePrimary;
