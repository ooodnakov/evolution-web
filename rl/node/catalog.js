/**
 * Stable, deterministic vocabulary for the RL observation and choice rows.
 *
 * One catalog per `rl/schema.json.schema_version`. Every registry namespace is
 * built from the actual gameplay registries (never from a deck, a concrete
 * game or creation order), sorted lexicographically and given ids starting at
 * 1 with 0 reserved for "none".
 *
 * Schema enum namespaces (entity kinds, decision kinds, phases, addons, ...)
 * keep the canonical order of `rl/schema.json`: consumers index Python
 * `Discrete` spaces with the schema order, and multi-binary fields with the
 * schema list order, so re-sorting those would silently remap values.
 */

import schema from '../schema.json';

import * as cards from '~/shared/models/game/evolution/cards/index';
import * as animalTraits from '~/shared/models/game/evolution/traitTypes';
import * as plantTypes from '~/shared/models/game/evolution/plantarium/plantTypes';
import * as plantTraits from '~/shared/models/game/evolution/plantarium/plantTraitTypes';
import {
  TRAIT_ANIMAL_FLAG,
  TRAIT_COOLDOWN_DURATION,
  TRAIT_COOLDOWN_LINK,
  TRAIT_COOLDOWN_PLACE,
} from '~/shared/models/game/evolution/constants';

const sortedUnique = (values) => Array.from(new Set(
  values.filter((value) => typeof value === 'string' && value.length > 0)
)).sort();

const cardTypeName = (entry) => (
  typeof entry === 'string'
    ? entry
    : (entry != null && typeof entry.type === 'string' ? entry.type : null)
);

/** Registry namespace: 0 = NONE, then lexicographic ids starting at 1. */
const registryNamespace = (names) => {
  const namespace = {NONE: 0};
  sortedUnique(names).forEach((name, index) => {
    namespace[name] = index + 1;
  });
  return Object.freeze(namespace);
};

/** Schema enum namespace: ids are the canonical `rl/schema.json` indices. */
const orderedNamespace = (names) => {
  const namespace = {};
  names.forEach((name, index) => {
    namespace[name] = index;
  });
  return Object.freeze(namespace);
};

const buildCatalog = () => {
  const animalTraitNames = sortedUnique(Object.values(animalTraits));
  const plantTraitNames = sortedUnique(Object.values(plantTraits));
  const plantTypeNames = sortedUnique(Object.values(plantTypes));
  const cardTypeNames = sortedUnique(Object.values(cards).map(cardTypeName));

  // Cooldown links are either an explicit link name (`EATING`, `TAKE_SHELL`)
  // or a trait type used as its own cooldown key, so the union of both
  // registries plus the link constants covers every key gameplay stores.
  const cooldownKeys = sortedUnique([
    ...animalTraitNames,
    ...plantTraitNames,
    ...Object.values(TRAIT_COOLDOWN_LINK),
  ]);

  const namespaces = {
    card_types: registryNamespace(cardTypeNames),
    trait_types: registryNamespace(animalTraitNames),
    plant_trait_types: registryNamespace(plantTraitNames),
    // Combined namespace used by observation trait rows and hand cards: a
    // single id space keeps plant and animal trait ids unambiguous.
    all_trait_types: registryNamespace([...animalTraitNames, ...plantTraitNames]),
    plant_types: registryNamespace(plantTypeNames),
    // Multi-binary bit order must stay identical to `schema.animal_flags`.
    animal_flag_bits: orderedNamespace(schema.animal_flags),
    animal_flags: registryNamespace(Object.values(TRAIT_ANIMAL_FLAG)),
    cooldown_keys: registryNamespace(cooldownKeys),
    cooldown_places: registryNamespace(Object.values(TRAIT_COOLDOWN_PLACE)),
    cooldown_durations: registryNamespace(Object.values(TRAIT_COOLDOWN_DURATION)),
    addons: orderedNamespace(schema.addons),
    phase_values: orderedNamespace(schema.phase_values),
    decision_kinds: orderedNamespace(schema.decision_kinds),
    entity_kinds: orderedNamespace(schema.entity_kinds),
    value_kinds: orderedNamespace(schema.value_kinds),
    feeding_source_types: orderedNamespace(schema.feeding_source_types),
    choice_field_kinds: orderedNamespace(schema.choice_field_kinds),
    choice_kinds: orderedNamespace(schema.choice_kinds),
    command_types: orderedNamespace(schema.command_types),
  };

  return Object.freeze({
    schema_version: schema.schema_version,
    engine_version: schema.engine_version,
    namespaces: Object.freeze(namespaces),
  });
};

let cachedCatalog = null;

/**
 * Return the process-wide catalog. Construction depends only on the imported
 * registries and `rl/schema.json`, so it is safe to memoize.
 */
export const createCatalog = () => {
  if (cachedCatalog === null) {
    cachedCatalog = buildCatalog();
  }
  return cachedCatalog;
};

/** Id of `name` inside `namespace`, or 0 for any unknown name/namespace. */
export const typeId = (catalog, namespace, name) => {
  if (catalog == null || catalog.namespaces == null) {
    throw new Error('INVALID_CATALOG');
  }
  if (typeof name !== 'string') return 0;
  const names = catalog.namespaces[namespace];
  if (names == null) return 0;
  return Object.prototype.hasOwnProperty.call(names, name) ? names[name] : 0;
};

// Eager, shared instance consumed by `rl/node/worker.cjs` hello handshake.
export const catalog = createCatalog();

export const getCatalog = createCatalog;

export default createCatalog;
