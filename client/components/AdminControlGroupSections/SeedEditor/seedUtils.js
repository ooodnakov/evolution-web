import {load as loadYaml, dump as dumpYaml} from 'js-yaml';

import * as cardsData from '~/shared/models/game/evolution/cards/index';
import * as plantTypes from '~/shared/models/game/evolution/plantarium/plantTypes';
import * as traitTypes from '~/shared/models/game/evolution/traitTypes/index';

export const defaultSeedString = `deck: 12 carnivorous, 6 sharp
phase: feeding
food: 2
players:
  - hand: 1 sharp, 1 camo
    continent: carn sharp, carn camo
  - hand: 1 sharp, 1 camo
    continent: carn sharp, carn camo
`;

const cardTypeKeys = Object.keys(cardsData).filter((key) => key.startsWith('Card'));
const plantTypeKeys = Object.keys(plantTypes);
const traitTypeKeys = Object.keys(traitTypes).filter((key) => key.startsWith('Trait'));

const findMatchingKey = (keys, name) => {
  if (!name) return '';
  const trimmed = String(name).trim();
  if (!trimmed) return '';
  const normalized = trimmed.toLowerCase();
  return keys.find((key) => key.toLowerCase() === normalized)
    || keys.find((key) => key.toLowerCase().includes(normalized))
    || trimmed;
};

const normalizeCardValue = (value) => findMatchingKey(cardTypeKeys, value);
const normalizePlantValue = (value) => findMatchingKey(plantTypeKeys, value);
const normalizeTraitValue = (value) => {
  if (!value) return '';
  const trimmed = String(value).trim();
  if (!trimmed) return '';
  const [baseWithValue, ...linkParts] = trimmed.split('$');
  const [baseName, ...valueParts] = baseWithValue.split('=');
  const traitKey = findMatchingKey(traitTypeKeys, baseName);
  if (!traitKey) return trimmed;
  const valueSuffix = valueParts.length ? `=${valueParts.join('=')}` : '';
  const linkSuffix = linkParts.length ? `$${linkParts.join('$')}` : '';
  return `${traitKey}${valueSuffix}${linkSuffix}`;
};

const toCardEntries = (deckString, normalizer = normalizeCardValue) => {
  if (!deckString) return [];
  return deckString
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const parts = entry.split(/\s+/);
      if (parts.length === 0) return {count: 1, card: ''};
      const first = parts[0];
      if (/^\d+$/.test(first)) {
        return {
          count: parseInt(first, 10),
          card: parts.slice(1).join(' ') || '',
        };
      }
      return {
        count: 1,
        card: entry,
      };
    })
    .map(({count, card}) => ({
      count,
      card: normalizer ? normalizer(card) : card,
    }));
};

const toDeckString = (entries, normalizer = normalizeCardValue) => entries
  .filter((entry) => entry.card && entry.card.trim())
  .map(({count, card}) => {
    const safeCount = Math.max(1, Number(count) || 1);
    const normalizedCard = normalizer ? normalizer(card) : card;
    return `${safeCount} ${String(normalizedCard || '').trim()}`;
  })
  .join(', ');

const toAnimals = (continentString) => {
  if (!continentString) return [];
  return continentString
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const parts = entry.split(/\s+/).filter(Boolean);
      let food = 0;
      const traits = [];
      parts.forEach((part) => {
        if (/^\++$/.test(part)) {
          food = part.length;
        } else {
          traits.push(normalizeTraitValue(part));
        }
      });
      return {food, traits};
    });
};

const toContinentString = (animals) => animals
  .filter((animal) => animal.traits.length > 0 || animal.food > 0)
  .map(({food, traits}) => {
    const foodPart = food > 0 ? '+'.repeat(food) + ' ' : '';
    const normalizedTraits = traits
      .map((trait) => normalizeTraitValue(trait))
      .filter(Boolean)
      .join(' ');
    return `${foodPart}${normalizedTraits}`.trim();
  })
  .filter(Boolean)
  .join(', ');

const ensurePlayer = (player) => ({
  hand: Array.isArray(player.hand) ? player.hand : [],
  animals: Array.isArray(player.animals) ? player.animals : [],
});

const makeSeedStateFromYaml = (seed) => {
  const deck = toCardEntries(seed.deck, normalizeCardValue);
  const deckPlants = toCardEntries(seed.deckPlants, normalizePlantValue);
  const players = Array.isArray(seed.players)
    ? seed.players.map((player) => ensurePlayer({
      hand: toCardEntries(player.hand, normalizeCardValue),
      animals: toAnimals(player.continent),
    }))
    : [];
  return {
    phase: seed.phase || 'deploy',
    food: Number(seed.food) || 0,
    deck,
    deckPlants,
    players,
    settings: seed.settings || {},
  };
};

const defaultSeedState = (() => {
  try {
    const parsed = loadYaml(defaultSeedString) || {};
    return makeSeedStateFromYaml(parsed);
  } catch (error) {
    return {
      phase: 'deploy',
      food: 0,
      deck: [],
      deckPlants: [],
      players: [],
      settings: {},
    };
  }
})();

export const parseSeedString = (seedString) => {
  try {
    const parsed = loadYaml(seedString || '') || {};
    return {
      state: makeSeedStateFromYaml(parsed),
      error: null,
    };
  } catch (error) {
    return {
      state: defaultSeedState,
      error,
    };
  }
};

export const buildSeedString = (state) => {
  const players = state.players.map(({hand, animals}) => ({
    hand: toDeckString(hand, normalizeCardValue),
    continent: toContinentString(animals),
  }));
  const payload = {
    deck: toDeckString(state.deck, normalizeCardValue),
    phase: state.phase || 'deploy',
    food: Number(state.food) || 0,
    players,
  };
  if (state.deckPlants && state.deckPlants.length > 0) {
    payload.deckPlants = toDeckString(state.deckPlants, normalizePlantValue);
  }
  if (state.settings && Object.keys(state.settings).length > 0) {
    payload.settings = state.settings;
  }
  return dumpYaml(payload, {lineWidth: 120});
};

export const mergePlayersWithRoom = (seedState, roomPlayerCount) => {
  if (!roomPlayerCount || roomPlayerCount <= seedState.players.length) return seedState;
  const players = [...seedState.players];
  for (let i = players.length; i < roomPlayerCount; i += 1) {
    players.push(ensurePlayer({hand: [], animals: []}));
  }
  return {...seedState, players};
};

export const humanizeKey = (key) => key
  .replace(/^Card/, '')
  .replace(/^Trait/, '')
  .replace(/([a-z])([A-Z])/g, '$1 $2')
  .replace(/_/g, ' ')
  .trim();
