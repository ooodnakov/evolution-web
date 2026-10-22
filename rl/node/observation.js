/**
 * Player-information observation projection.
 *
 * Produces exactly the structure documented by `rl/schema.json.observation`.
 * The input `game` is an authoritative GameModel; this module never mutates it
 * and never reads the clock, RNG or id counter. Everything that leaves this
 * module is already restricted to what `agentId` may know:
 *
 *  - opponents' hidden traits are omitted;
 *  - opponents' transient trait values are omitted;
 *  - hidden plant traits are replaced by the inherent capabilities of the
 *    public plant type (PlantCarnivorous);
 *  - deck order, opponent hands, question/hunt internals, settings.seed and
 *    UUIDs never appear.
 *
 * `view` is the current decision projection built by the environment:
 *
 *   {
 *     actor: string|null,
 *     decisionKind: 'NORMAL'|'DEFENSE'|...,
 *     choiceRows: [choiceRow],
 *     mask: number[],
 *     source: {kind, slot}|null,
 *     target: {kind, slot}|null,
 *     attackTraitType: string|number|null,
 *     prefix: [choiceRow],
 *   }
 *
 * A choice row may use either camelCase or schema snake_case keys. `typeId`
 * values supplied by the action enumerator are used verbatim (they already
 * reference the right catalog namespace); a `type` name is resolved through
 * the shared type namespaces.
 */

import {entitiesOfGame} from './slots';
import {createCatalog, typeId} from './catalog';
import {TraitFatTissue} from '~/shared/models/game/evolution/traitTypes';
import {PlantCarnivorous} from '~/shared/models/game/evolution/plantarium/plantTypes';
import {
  PlantTraitHiddenCarnivorous,
  PlantTraitHiddenIntellect,
} from '~/shared/models/game/evolution/plantarium/plantTraitTypes';

const OBSERVATION_SCHEMA = 'OBSERVATION_SCHEMA';

const fail = () => {
  throw new Error(OBSERVATION_SCHEMA);
};

const pick = (...values) => values.find((value) => value !== undefined);

const noneValue = () => ({
  value_kind: 0,
  value_number: 0,
  value_entity_kind: 0,
  value_entity_slot: -1,
  value_trait_slot: -1,
  source_type: 0,
  source_slot: -1,
  auto_share: 0,
});

export const createObservation = (options = {}) => {
  const catalog = options.catalog || createCatalog();

  const requireTypeId = (namespace, name) => {
    const names = catalog.namespaces[namespace];
    if (names == null || typeof name !== 'string'
      || !Object.prototype.hasOwnProperty.call(names, name)) {
      fail();
    }
    return names[name];
  };

  const entityKindId = (kind) => {
    if (Number.isInteger(kind)) {
      if (kind < 0 || kind >= Object.keys(catalog.namespaces.entity_kinds).length) fail();
      return kind;
    }
    if (kind == null) return 0;
    if (typeof kind !== 'string') fail();
    const names = catalog.namespaces.entity_kinds;
    if (!Object.prototype.hasOwnProperty.call(names, kind)) fail();
    return names[kind];
  };

  const normalizeRef = (ref) => {
    if (ref == null) return {kind: 0, slot: -1};
    if (typeof ref !== 'object' || Array.isArray(ref)) fail();
    const kind = entityKindId(pick(ref.kind, ref.entityKind));
    const rawSlot = pick(ref.slot, ref.entitySlot);
    const slot = rawSlot === undefined ? -1 : rawSlot;
    if (!Number.isInteger(slot)) fail();
    return {kind, slot};
  };

  const choiceTypeId = (row) => {
    const rawId = pick(row.typeId, row.type_id);
    if (Number.isInteger(rawId)) return rawId;
    const name = pick(row.typeName, row.type_name, row.type, row.name);
    if (typeof name !== 'string') return 0;
    const explicitNamespace = pick(row.typeNamespace, row.type_namespace, row.namespace);
    const candidates = explicitNamespace
      ? [explicitNamespace]
      : ['command_types', 'card_types', 'all_trait_types', 'plant_types'];
    for (const namespace of candidates) {
      const id = typeId(catalog, namespace, name);
      if (id !== 0) return id;
    }
    return 0;
  };

  const integerField = (value, fallback) => {
    if (value === undefined || value === null) return fallback;
    if (!Number.isInteger(value)) fail();
    return value;
  };

  const serializeChoiceRow = (row) => {
    if (row == null || typeof row !== 'object' || Array.isArray(row)) fail();
    const fieldKind = pick(row.fieldKind, row.field_kind);
    const choiceKind = pick(row.choiceKind, row.choice_kind);
    if (fieldKind === undefined || choiceKind === undefined) fail();
    const entitySlot = pick(row.entitySlot, row.entity_slot);
    const traitSlot = pick(row.traitSlot, row.trait_slot);
    const scalar = pick(row.scalar, row.value);
    return {
      field_kind: requireTypeId('choice_field_kinds', fieldKind),
      choice_kind: requireTypeId('choice_kinds', choiceKind),
      type_id: choiceTypeId(row),
      entity_kind: entityKindId(pick(row.entityKind, row.entity_kind)),
      entity_slot: integerField(entitySlot, -1),
      trait_slot: integerField(traitSlot, -1),
      scalar: integerField(scalar, 0),
    };
  };

  const observe = (game, agentId, view = {}) => {
    if (game == null || typeof game !== 'object') fail();
    if (typeof agentId !== 'string' || agentId.length === 0) fail();
    if (view == null || typeof view !== 'object') fail();

    const agent = game.getPlayer(agentId);
    const agentSeat = agent ? agent.index : -1;

    const actor = view.actor === undefined ? agentId : view.actor;
    const isActor = actor !== null && actor !== undefined && actor === agentId;

    const decisionKindName = pick(view.decisionKind, view.kind) || 'NORMAL';
    const decisionKind = requireTypeId('decision_kinds', decisionKindName);

    const entities = entitiesOfGame(game);

    // ---- local trait projection ------------------------------------------
    const animalEntriesById = new Map();
    const plantEntriesById = new Map();
    const exposedAnimalTraits = new Map();
    const exposedPlantTraits = new Map();
    // Visible trait id -> local reference, used to encode TRAIT_REF values.
    const visibleTraitIndex = new Map();

    entities.animals.forEach((entry) => {
      const own = entry.ownerId === agentId;
      const traits = Array.from(entry.animal.traits.values())
        .filter((trait) => own || !trait.getDataModel().hidden);
      exposedAnimalTraits.set(entry.id, traits);
      traits.forEach((trait, traitSlot) => {
        visibleTraitIndex.set(trait.id, {
          kind: 1,
          entitySlot: entry.slot,
          traitSlot,
        });
      });
      animalEntriesById.set(entry.id, entry);
    });

    entities.plants.forEach((entry) => {
      const traits = Array.from(entry.plant.traits.values())
        .filter((trait) => !trait.getDataModel().hidden);
      exposedPlantTraits.set(entry.id, traits);
      traits.forEach((trait, traitSlot) => {
        visibleTraitIndex.set(trait.id, {
          kind: 2,
          entitySlot: entry.slot,
          traitSlot,
        });
      });
      plantEntriesById.set(entry.id, entry);
    });

    const linkedRef = (trait) => {
      if (trait.linkId == null) return {kind: 0, slot: -1, traitSlot: -1};
      if (animalEntriesById.has(trait.linkAnimalId)) {
        const entry = animalEntriesById.get(trait.linkAnimalId);
        const traits = exposedAnimalTraits.get(entry.id) || [];
        const traitSlot = traits.findIndex((candidate) => candidate.id === trait.linkId);
        return {kind: 1, slot: entry.slot, traitSlot: traitSlot < 0 ? -1 : traitSlot};
      }
      if (plantEntriesById.has(trait.linkAnimalId)) {
        const entry = plantEntriesById.get(trait.linkAnimalId);
        const traits = exposedPlantTraits.get(entry.id) || [];
        const traitSlot = traits.findIndex((candidate) => candidate.id === trait.linkId);
        return {kind: 2, slot: entry.slot, traitSlot: traitSlot < 0 ? -1 : traitSlot};
      }
      return {kind: 0, slot: -1, traitSlot: -1};
    };

    const encodeTraitValue = (value) => {
      if (value === undefined || value === null || value === false) {
        return noneValue();
      }
      if (value === true) {
        return {...noneValue(), value_kind: 1, value_number: 1};
      }
      if (typeof value === 'number') {
        if (!Number.isFinite(value) || !Number.isInteger(value)) fail();
        return {...noneValue(), value_kind: 2, value_number: value};
      }
      if (typeof value === 'string') {
        const traitRef = visibleTraitIndex.get(value);
        if (traitRef) {
          return {
            ...noneValue(),
            value_kind: 3,
            value_entity_kind: traitRef.kind,
            value_entity_slot: traitRef.entitySlot,
            value_trait_slot: traitRef.traitSlot,
          };
        }
        const plantRef = plantEntriesById.get(value);
        if (plantRef) {
          return {
            ...noneValue(),
            value_kind: 4,
            value_entity_kind: 2,
            value_entity_slot: plantRef.slot,
          };
        }
        // A raw id that is not part of the visible projection must never be
        // stringified into the observation; an unresolvable reference is "none".
        return noneValue();
      }
      if (typeof value === 'object' && !Array.isArray(value)) {
        const sourceType = value.sourceType;
        if (typeof sourceType === 'string') {
          let sourceSlot = -1;
          if (sourceType === 'PLANT') {
            const plantRef = plantEntriesById.get(value.sourceId);
            sourceSlot = plantRef ? plantRef.slot : -1;
          }
          return {
            ...noneValue(),
            value_kind: 5,
            source_type: typeId(catalog, 'feeding_source_types', sourceType),
            source_slot: sourceSlot,
            auto_share: value.autoShare ? 1 : 0,
          };
        }
        return fail();
      }
      return fail();
    };

    const cooldownRows = (gameState, places) => {
      const rows = [];
      const cooldowns = gameState.cooldowns;
      if (cooldowns == null) return rows;
      places.forEach(([place, placeId]) => {
        if (placeId == null) return;
        const entries = cooldowns.getIn([place, placeId]);
        if (entries == null) return;
        entries.forEach((duration, link) => {
          rows.push({
            key: requireTypeId('cooldown_keys', link),
            remaining: Number.isInteger(duration) ? duration : 0,
          });
        });
      });
      rows.sort((left, right) => left.key - right.key || left.remaining - right.remaining);
      return rows;
    };

    const buildTraitRow = (trait, hideValue, places) => {
      const linked = linkedRef(trait);
      return {
        type: requireTypeId('all_trait_types', trait.type),
        disabled: !!trait.disabled,
        covers: Number.isInteger(trait.covers) ? trait.covers : 0,
        link_source: trait.linkSource === true ? 1 : 0,
        linked_entity_kind: linked.kind,
        linked_entity_slot: linked.slot,
        linked_trait_slot: linked.traitSlot,
        ...(hideValue ? noneValue() : encodeTraitValue(trait.value)),
        cooldowns: cooldownRows(game, places),
      };
    };

    const neutralTraitRow = (traitType) => ({
      type: requireTypeId('all_trait_types', traitType),
      disabled: false,
      covers: 0,
      link_source: 0,
      linked_entity_kind: 0,
      linked_entity_slot: -1,
      linked_trait_slot: -1,
      ...noneValue(),
      cooldowns: [],
    });

    // ---- global ----------------------------------------------------------
    const seatOf = (playerId) => {
      if (playerId == null) return -1;
      const player = game.getPlayer(playerId);
      return player ? player.index : -1;
    };

    const addonNames = Object.keys(catalog.namespaces.addons);
    const settings = game.settings || {};
    const global = {
      phase: requireTypeId('phase_values', game.status.phase),
      turn: game.status.turn,
      round: game.status.round,
      seat: agentSeat,
      current_player: seatOf(game.status.currentPlayer),
      round_player: seatOf(game.status.roundPlayer),
      food: typeof game.getFood === 'function' ? game.getFood() : game.food,
      deck_count: game.deck == null ? 0 : game.deck.size,
      active_players: game.getActualPlayers().size,
      addons: addonNames.map((name) => (settings[`addon_${name}`] ? 1 : 0)),
      half_deck: !!settings.halfDeck,
    };

    // ---- players ---------------------------------------------------------
    const players = game.players.toList().map((player) => ({
      seat: player.index,
      playing: player.playing,
      ended: player.ended,
      acted: player.acted,
      hand_count: player.hand.size,
      animal_count: player.continent.size,
    })).toArray();

    // ---- own hand --------------------------------------------------------
    const hand = !agent ? [] : agent.hand.map((card) => ({
      card_type: requireTypeId('card_types', card.type),
      trait1_type: card.trait1 == null ? 0 : requireTypeId('all_trait_types', card.trait1),
      trait2_type: card.trait2 == null ? 0 : requireTypeId('all_trait_types', card.trait2),
    })).toArray();

    // ---- animals ---------------------------------------------------------
    const flagNames = Object.keys(catalog.namespaces.animal_flag_bits);

    const animals = entities.animals.map((entry) => {
      const animal = entry.animal;
      const own = entry.ownerId === agentId;
      const traits = exposedAnimalTraits.get(animal.id) || [];

      let foodSize = 1;
      let fatSize = 0;
      traits.forEach((trait) => {
        if (trait.disabled) return;
        const data = trait.getDataModel();
        if (typeof data.food === 'number') foodSize += data.food;
        if (trait.type === TraitFatTissue) fatSize += 1;
      });
      foodSize = Math.max(1, foodSize);

      const placesFor = (trait) => [
        ['TRAIT', trait.id],
        ['ANIMAL', animal.id],
        ['PLAYER', animal.ownerId],
        ['GAME', 'default'],
      ];

      return {
        owner: entry.ownerSeat,
        position: entry.position,
        food: animal.getFood(),
        food_size: foodSize,
        fat_size: fatSize,
        flags: flagNames.map((name) => (animal.hasFlag(name) ? 1 : 0)),
        traits: traits.map((trait) => buildTraitRow(
          trait,
          !own && !!trait.getDataModel().transient,
          placesFor(trait)
        )),
      };
    });

    // ---- plants ----------------------------------------------------------
    const plants = entities.plants.map((entry) => {
      const plant = entry.plant;
      const traits = (exposedPlantTraits.get(plant.id) || []).map((trait) => buildTraitRow(
        trait,
        false,
        [
          ['TRAIT', trait.id],
          ['PLANT', plant.id],
          ['GAME', 'default'],
        ]
      ));
      // PlantCarnivorous is public type information: its hidden capabilities
      // are re-derived instead of exposing hidden dynamic trait state.
      if (plant.type === PlantCarnivorous) {
        traits.push(neutralTraitRow(PlantTraitHiddenCarnivorous));
        traits.push(neutralTraitRow(PlantTraitHiddenIntellect));
      }
      return {
        type: requireTypeId('plant_types', plant.type),
        food: plant.getFood(),
        covers: plant.getCovers(),
        cover_slots: plant.coverSlots,
        fruit: plant.isFruit() ? 1 : 0,
        traits,
      };
    });

    // ---- shells ----------------------------------------------------------
    const shells = entities.shells.map((entry) => ({
      type: requireTypeId('all_trait_types', entry.shell.type),
      value: entry.shell.value === true ? 1 : 0,
      slot: entry.slot,
    }));

    // ---- decision context / choices -------------------------------------
    const source = normalizeRef(view.source);
    const target = normalizeRef(view.target);

    let attackTraitType = 0;
    const rawAttack = pick(view.attackTraitType, view.attack_trait_type);
    if (Number.isInteger(rawAttack)) {
      attackTraitType = rawAttack;
    } else if (typeof rawAttack === 'string') {
      attackTraitType = typeId(catalog, 'all_trait_types', rawAttack);
    }

    const rawChoices = pick(view.choiceRows, view.choices) || [];
    const rawPrefix = view.prefix || [];
    if (!Array.isArray(rawChoices) || !Array.isArray(rawPrefix)) fail();

    let choices = [];
    let actionMask = [];
    if (isActor) {
      choices = rawChoices.map(serializeChoiceRow);
      const rawMask = pick(view.mask, view.actionMask);
      if (rawMask === undefined || rawMask === null) {
        actionMask = choices.map(() => 1);
      } else {
        if (!Array.isArray(rawMask) || rawMask.length !== choices.length) fail();
        actionMask = rawMask.map((value) => (value ? 1 : 0));
      }
    }

    const context = {
      kind: decisionKind,
      source_kind: source.kind,
      source_slot: source.slot,
      target_kind: target.kind,
      target_slot: target.slot,
      attack_trait_type: attackTraitType,
      // Prefix rows are the acting player's own choices only.
      prefix: isActor ? rawPrefix.map(serializeChoiceRow) : [],
    };

    return {
      global,
      players,
      hand,
      animals,
      plants,
      shells,
      context,
      choices,
      action_mask: actionMask,
    };
  };

  return {observe};
};

export default createObservation;
