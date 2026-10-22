import {expect} from 'chai';

import * as cards from '~/shared/models/game/evolution/cards/index';
import * as animalTraits from '~/shared/models/game/evolution/traitTypes';
import * as plantTypes from '~/shared/models/game/evolution/plantarium/plantTypes';
import * as plantTraits from '~/shared/models/game/evolution/plantarium/plantTraitTypes';
import {
  TRAIT_ANIMAL_FLAG,
  TRAIT_COOLDOWN_LINK,
} from '~/shared/models/game/evolution/constants';
import schema from '../../schema.json';

import {createCatalog, typeId} from '../catalog';

const REGISTRY_NAMESPACES = [
  'card_types',
  'trait_types',
  'plant_trait_types',
  'all_trait_types',
  'plant_types',
  'animal_flags',
  'cooldown_keys',
];

const registryNames = (namespace) => Object.keys(namespace).filter((name) => name !== 'NONE');

describe('rl/node/catalog', () => {
  const catalog = createCatalog();

  it('reserves id 0 for NONE and starts registry ids at 1', () => {
    REGISTRY_NAMESPACES.forEach((namespace) => {
      const names = catalog.namespaces[namespace];
      expect(names.NONE, namespace).to.equal(0);
      const namesOnly = registryNames(names);
      expect(namesOnly.length, namespace).to.be.greaterThan(0);
      namesOnly.forEach((name, index) => {
        expect(names[name], `${namespace}.${name}`).to.equal(index + 1);
      });
    });
  });

  it('sorts every registry namespace lexicographically', () => {
    REGISTRY_NAMESPACES.forEach((namespace) => {
      const names = registryNames(catalog.namespaces[namespace]);
      expect(names, namespace).to.deep.equal([...names].sort());
    });
  });

  it('collects every registry export, not a deck subset', () => {
    expect(registryNames(catalog.namespaces.card_types)).to.have.lengthOf(Object.keys(cards).length);
    expect(registryNames(catalog.namespaces.trait_types)).to.have.lengthOf(Object.keys(animalTraits).length);
    expect(registryNames(catalog.namespaces.plant_trait_types)).to.have.lengthOf(Object.keys(plantTraits).length);
    expect(registryNames(catalog.namespaces.plant_types)).to.have.lengthOf(Object.keys(plantTypes).length);
    expect(registryNames(catalog.namespaces.animal_flags)).to.have.lengthOf(Object.keys(TRAIT_ANIMAL_FLAG).length);
    expect(registryNames(catalog.namespaces.all_trait_types)).to.have.lengthOf(
      new Set([...Object.values(animalTraits), ...Object.values(plantTraits)]).size
    );
    Object.values(cards).forEach((card) => {
      expect(card.type).to.be.a('string');
      expect(typeId(catalog, 'card_types', card.type), card.type).to.be.at.least(1);
    });
    Object.values(animalTraits).forEach((name) => {
      expect(typeId(catalog, 'trait_types', name), name).to.be.at.least(1);
    });
    Object.values(plantTraits).forEach((name) => {
      expect(typeId(catalog, 'plant_trait_types', name), name).to.be.at.least(1);
    });
    Object.values(plantTypes).forEach((name) => {
      expect(typeId(catalog, 'plant_types', name), name).to.be.at.least(1);
    });
  });

  it('keeps canonical animal flag bit order separate from sorted ids', () => {
    const bits = catalog.namespaces.animal_flag_bits;
    expect(Object.keys(bits)).to.deep.equal(schema.animal_flags);
    schema.animal_flags.forEach((name, index) => {
      expect(bits[name], name).to.equal(index);
    });
    const sorted = catalog.namespaces.animal_flags;
    expect(registryNames(sorted)).to.deep.equal(
      Object.values(TRAIT_ANIMAL_FLAG).sort()
    );
    Object.values(TRAIT_ANIMAL_FLAG).forEach((name) => {
      expect(sorted).to.have.property(name);
    });
  });

  it('covers every gameplay cooldown key', () => {
    Object.values(TRAIT_COOLDOWN_LINK).forEach((name) => {
      expect(typeId(catalog, 'cooldown_keys', name), name).to.be.at.least(1);
    });
    ['TraitCarnivorous', 'TraitWaiter', 'PlantTraitParasiticLink'].forEach((name) => {
      expect(typeId(catalog, 'cooldown_keys', name), name).to.be.at.least(1);
    });
    ['EATING', 'TAKE_SHELL'].forEach((name) => {
      expect(typeId(catalog, 'cooldown_keys', name), name).to.be.at.least(1);
    });
  });

  it('maps schema enums to their canonical schema indices', () => {
    schema.entity_kinds.forEach((name, index) => {
      expect(typeId(catalog, 'entity_kinds', name), name).to.equal(index);
    });
    schema.decision_kinds.forEach((name, index) => {
      expect(typeId(catalog, 'decision_kinds', name), name).to.equal(index);
    });
    schema.value_kinds.forEach((name, index) => {
      expect(typeId(catalog, 'value_kinds', name), name).to.equal(index);
    });
    schema.command_types.forEach((name, index) => {
      expect(typeId(catalog, 'command_types', name), name).to.equal(index);
    });
    schema.addons.forEach((name, index) => {
      expect(typeId(catalog, 'addons', name), name).to.equal(index);
    });
  });

  it('is memoized and stable across calls', () => {
    const again = createCatalog();
    expect(again).to.equal(catalog);
    expect(JSON.parse(JSON.stringify(again))).to.deep.equal(JSON.parse(JSON.stringify(catalog)));
  });

  it('is independent of Math.random and Date.now', () => {
    const originalRandom = Math.random;
    const originalNow = Date.now;
    const catalogPath = require.resolve('../catalog');
    const cachedModule = require.cache[catalogPath];
    let fresh;
    Math.random = () => {
      throw new Error('catalog used Math.random');
    };
    Date.now = () => {
      throw new Error('catalog used Date.now');
    };
    try {
      delete require.cache[catalogPath];
      fresh = require('../catalog');
      expect(fresh.createCatalog()).to.deep.equal(catalog);
      expect(fresh.createCatalog()).to.equal(fresh.createCatalog());
    } finally {
      Math.random = originalRandom;
      Date.now = originalNow;
      require.cache[catalogPath] = cachedModule;
    }
    expect(JSON.parse(JSON.stringify(fresh.createCatalog()))).to.deep.equal(JSON.parse(JSON.stringify(catalog)));
  });

  it('returns 0 for unknown names and namespaces', () => {
    expect(typeId(catalog, 'trait_types', 'TraitNotInTheGame')).to.equal(0);
    expect(typeId(catalog, 'missing_namespace', 'TraitSwimming')).to.equal(0);
    expect(typeId(catalog, 'trait_types', null)).to.equal(0);
  });
});
