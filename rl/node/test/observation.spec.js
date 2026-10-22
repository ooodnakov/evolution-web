import {expect} from 'chai';
import fs from 'fs';
import path from 'path';
import {List, OrderedMap} from 'immutable';

import {createRuntime} from '../runtime';
import {withGameRuntime} from '~/shared/utils/gameRuntime';
import {GameModel, PHASE, AmbushRecord} from '~/shared/models/game/GameModel';
import {SettingsRecord} from '~/shared/models/game/GameSettings';
import {AnimalModel} from '~/shared/models/game/evolution/AnimalModel';
import {TraitModel} from '~/shared/models/game/evolution/TraitModel';
import {CardModel} from '~/shared/models/game/CardModel';
import PlantModel from '~/shared/models/game/evolution/plantarium/PlantModel';
import {PlantCarnivorous} from '~/shared/models/game/evolution/plantarium/plantTypes';
import {
  PlantTraitHiddenCarnivorous,
  PlantTraitHiddenIntellect,
} from '~/shared/models/game/evolution/plantarium/plantTraitTypes';

import {createCatalog, typeId} from '../catalog';
import {createObservation} from '../observation';

const schema = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../schema.json'), 'utf8'));
const catalog = createCatalog();
const {observe} = createObservation({catalog});

const traitId = (name) => typeId(catalog, 'all_trait_types', name);
const traitRowOf = (animalRow, name) => animalRow.traits.find((row) => row.type === traitId(name));

const traitSpec = (trait) => (
  typeof trait === 'string' ? {type: trait, value: undefined} : trait
);

/**
 * Build a real GameModel for one or two seats. Every entity id is allocated
 * inside a seeded runtime, so two builds never share UUIDs.
 */
const buildGame = (options = {}) => withGameRuntime(
  createRuntime({seed: options.seed === undefined ? 1 : options.seed}),
  () => {
    const settings = new SettingsRecord({
      maxPlayers: 2,
      randomPlayers: false,
      halfDeck: false,
      maxCards: null,
      addon_base2: false,
      addon_timeToFly: false,
      addon_continents: false,
      addon_bonus: false,
      addon_plantarium: options.plantarium !== false,
      addon_customff: false,
      addon_lifecycle: false,
      seed: options.settingsSeed === undefined ? null : options.settingsSeed,
    });
    const room = {id: 'room-rl', users: List(['p0', 'p1']), settings};
    let game = GameModel.new(room);

    const ids = {};
    let plant = null;
    if (options.withPlant !== false) {
      plant = PlantModel.new(PlantCarnivorous);
      game = game.setIn(['plants', plant.id], plant);
      ids.plant = plant.id;
    }

    if (options.withShell !== false) {
      const shell = TraitModel.new('TraitShell');
      game = game.setIn(['areas', 'STANDARD', 'shells', shell.id], shell);
      ids.shell = shell.id;
    }

    const makeHand = (types) => List(types.map((type) => CardModel.new(type)));
    game = game.setIn(
      ['players', 'p0', 'hand'],
      makeHand(options.p0Hand || ['CardSwimming'])
    );
    game = game.setIn(
      ['players', 'p1', 'hand'],
      makeHand(options.p1Hand || ['CardCarnivorous'])
    );

    const addAnimal = (ownerId, traits) => {
      let animal = AnimalModel.new(ownerId);
      (traits || []).forEach((rawTrait) => {
        const spec = traitSpec(rawTrait);
        let trait = TraitModel.new(spec.type);
        if (spec.value !== undefined) trait = trait.set('value', spec.value);
        animal = animal.traitAttach(trait);
      });
      if (options.flags && options.flags[ownerId]) {
        Object.keys(options.flags[ownerId]).forEach((flag) => {
          animal = animal.set('flags', animal.flags.set(flag, true));
        });
      }
      game = game.setIn(['players', ownerId, 'continent', animal.id], animal);
      return animal;
    };

    const p0Animal = options.p0Animals === false
      ? null
      : addAnimal('p0', options.p0Traits || ['TraitCarnivorous']);
    const p1Animal = options.p1Animals === false
      ? null
      : addAnimal('p1', options.p1Traits || ['TraitSwimming']);
    if (p0Animal) ids.p0Animal = p0Animal.id;
    if (p1Animal) ids.p1Animal = p1Animal.id;

    if (options.ambush) {
      game = game.set('ambush', new AmbushRecord({
        ambushers: OrderedMap(options.ambush),
        targetPlayerId: 'p1',
      }));
    }

    game = game
      .setIn(['status', 'phase'], PHASE.FEEDING)
      .setIn(['status', 'turn'], 3)
      .setIn(['status', 'round'], 1)
      .set('food', 5);

    return {game, ids, plant};
  }
);

const choiceRow = (overrides = {}) => ({
  fieldKind: 'TYPE',
  choiceKind: 'TYPE',
  typeId: typeId(catalog, 'command_types', 'gameEndTurnRequest'),
  entityKind: 'NONE',
  entitySlot: -1,
  traitSlot: -1,
  scalar: 0,
  ...overrides,
});

const makeView = (overrides = {}) => ({
  actor: 'p0',
  decisionKind: 'NORMAL',
  choiceRows: [
    choiceRow(),
    choiceRow({
      fieldKind: 'HAND_CARD',
      choiceKind: 'SCALAR',
      typeId: typeId(catalog, 'card_types', 'CardSwimming'),
    }),
  ],
  mask: [1, 0],
  source: {kind: 'ANIMAL', slot: 0},
  target: {kind: 'PLANT', slot: 0},
  attackTraitType: 'TraitCarnivorous',
  prefix: [choiceRow()],
  ...overrides,
});

// ---- local structural validator -----------------------------------------

const validateNode = (node, value, label) => {
  if (typeof node === 'string') {
    return validateNode(schema[node], value, label);
  }
  if (node.enum !== undefined) {
    const values = schema[node.enum];
    expect(Number.isInteger(value), `${label} enum value`).to.equal(true);
    expect(value, `${label} enum range`).to.be.within(0, values.length - 1);
    return;
  }
  if (node.int !== undefined) {
    expect(Number.isInteger(value), `${label} int`).to.equal(true);
    expect(value, `${label} int min`).to.be.at.least(-1);
    return;
  }
  if (node.bool !== undefined) {
    // MultiBinary(1): a one-element bit array on the wire.
    expect(value, `${label} bool bits`).to.be.an('array').with.lengthOf(1);
    expect([0, 1], `${label} bool bit`).to.include(value[0]);
    return;
  }
  if (node.int8 !== undefined) {
    expect([0, 1], `${label} int8`).to.include(value);
    return;
  }
  if (node.multibinary !== undefined) {
    const size = schema[node.multibinary].length;
    expect(value, `${label} multibinary array`).to.be.an('array').with.lengthOf(size);
    value.forEach((bit, index) => {
      expect([0, 1], `${label}[${index}]`).to.include(bit);
    });
    return;
  }
  if (node.sequence !== undefined) {
    expect(value, `${label} sequence`).to.be.an('array');
    value.forEach((item, index) => validateNode(node.sequence, item, `${label}[${index}]`));
    return;
  }
  expect(value, `${label} object`).to.be.an('object');
  expect(Object.keys(value), `${label} keys`).to.deep.equal(Object.keys(node));
  Object.keys(node).forEach((key) => validateNode(node[key], value[key], `${label}.${key}`));
};

const expectSchemaConformant = (observation) => {
  validateNode(schema.observation, observation, 'observation');
};

describe('rl/node/observation', () => {
  describe('schema shape', () => {
    it('passes the structural validator for the acting player', () => {
      const {game} = buildGame();
      const observation = observe(game, 'p0', makeView());
      expectSchemaConformant(observation);
      expect(Object.keys(observation)).to.deep.equal([
        'global',
        'players',
        'hand',
        'animals',
        'plants',
        'shells',
        'context',
        'choices',
        'action_mask',
      ]);
    });

    it('emits empty choices for an inactive player and for terminal frames', () => {
      const {game} = buildGame();
      const inactive = observe(game, 'p1', makeView());
      expectSchemaConformant(inactive);
      expect(inactive.choices).to.deep.equal([]);
      expect(inactive.action_mask).to.deep.equal([]);
      expect(inactive.context.prefix).to.deep.equal([]);

      const terminal = observe(game, 'p0', {
        actor: null,
        decisionKind: 'TERMINAL',
        choiceRows: [],
        mask: [],
        prefix: [],
      });
      expectSchemaConformant(terminal);
      expect(terminal.context.kind).to.equal(typeId(catalog, 'decision_kinds', 'TERMINAL'));
      expect(terminal.choices).to.deep.equal([]);
      expect(terminal.action_mask).to.deep.equal([]);
    });

    it('serializes choice rows with exactly the documented fields', () => {
      const {game} = buildGame();
      const observation = observe(game, 'p0', makeView({
        choiceRows: [{
          ...choiceRow(),
          cardId: 'card_1',
          animalId: 'animal_1',
          traitId: 'trait_1',
        }],
        mask: [1],
      }));
      expect(Object.keys(observation.choices[0])).to.deep.equal(
        Object.keys(schema.choice_row)
      );
      expect(observation.choices[0]).to.not.have.property('cardId');
      expect(observation.choices[0]).to.not.have.property('animalId');
      expect(observation.choices[0]).to.not.have.property('traitId');
      expect(observation.action_mask).to.deep.equal([1]);
    });

    it('projects global/player/hand fields', () => {
      const {game} = buildGame({
        p0Hand: ['CardSwimming', 'CardCarnivorous'],
        settingsSeed: 7,
      });
      const observation = observe(game, 'p0', makeView());
      expect(observation.global.phase).to.equal(typeId(catalog, 'phase_values', 'FEEDING'));
      expect(observation.global.turn).to.equal(3);
      expect(observation.global.round).to.equal(1);
      expect(observation.global.seat).to.equal(0);
      expect(observation.global.current_player).to.equal(0);
      expect(observation.global.round_player).to.equal(0);
      expect(observation.global.deck_count).to.equal(game.deck.size);
      expect(observation.global.active_players).to.equal(2);
      expect(observation.global.addons).to.deep.equal(
        schema.addons.map((name) => (name === 'plantarium' ? 1 : 0))
      );
      expect(observation.global.half_deck).to.deep.equal([0]);

      expect(observation.players).to.have.lengthOf(2);
      expect(observation.players[0]).to.deep.equal({
        seat: 0,
        playing: [1],
        ended: [0],
        acted: [0],
        hand_count: 2,
        animal_count: 1,
      });

      expect(observation.hand).to.deep.equal([
        {
          card_type: typeId(catalog, 'card_types', 'CardSwimming'),
          trait1_type: traitId('TraitSwimming'),
          trait2_type: 0,
        },
        {
          card_type: typeId(catalog, 'card_types', 'CardCarnivorous'),
          trait1_type: traitId('TraitCarnivorous'),
          trait2_type: 0,
        },
      ]);

      const opponent = observe(game, 'p1', makeView());
      expect(opponent.hand).to.deep.equal([{
        card_type: typeId(catalog, 'card_types', 'CardCarnivorous'),
        trait1_type: traitId('TraitCarnivorous'),
        trait2_type: 0,
      }]);
    });

    it('keeps an empty continent alive and reports zero animals', () => {
      const {game} = buildGame({p1Animals: false});
      const observation = observe(game, 'p0', makeView());
      expect(observation.players[1].animal_count).to.equal(0);
      expect(observation.players[1].playing).to.deep.equal([1]);
      expect(observation.animals).to.have.lengthOf(1);
    });

    it('publishes animal flags, plants and shells', () => {
      const {game} = buildGame({flags: {p0: {SHELL: true, HIBERNATED: true}}});
      const observation = observe(game, 'p0', makeView());
      const owner = observation.animals[0];
      const flagIndex = (name) => schema.animal_flags.indexOf(name);
      expect(owner.flags[flagIndex('SHELL')]).to.equal(1);
      expect(owner.flags[flagIndex('HIBERNATED')]).to.equal(1);
      expect(owner.flags[flagIndex('POISONED')]).to.equal(0);
      expect(observation.shells).to.deep.equal([{
        type: traitId('TraitShell'),
        value: 0,
        slot: 0,
      }]);
      expect(observation.plants).to.have.lengthOf(1);
      expect(observation.plants[0].type).to.equal(
        typeId(catalog, 'plant_types', 'PlantCarnivorous')
      );
      expect(observation.plants[0].fruit).to.equal(1);
      expect(observation.plants[0].food).to.equal(2);
    });
  });

  describe('privacy', () => {
    it('is identical for paired states differing only in private state', () => {
      const left = buildGame({
        seed: 11,
        settingsSeed: 111,
        p1Hand: ['CardSwimming', 'CardCarnivorous'],
        p1Traits: ['TraitSwimming', {type: 'TraitAnglerfish', value: true}],
        ambush: [['p1', true]],
      });
      const right = buildGame({
        seed: 22,
        settingsSeed: 222,
        p1Hand: ['CardCarnivorous', 'CardSwimming'],
        p1Traits: ['TraitSwimming', {type: 'TraitAnglerfish', value: false}],
        ambush: [['p1', false]],
      });
      const view = makeView();
      const leftGame = left.game;
      const rightGame = right.game
        .set('id', 'game_other_process')
        .setIn(
          ['players', 'p1', 'hand'],
          right.game.getIn(['players', 'p1', 'hand']).map((card) => card.set('id', `card_other_${card.id}`))
        );
      const leftObservation = observe(leftGame, 'p0', view);
      const rightObservation = observe(rightGame, 'p0', view);
      expect(leftObservation).to.deep.equal(rightObservation);
      expect(JSON.stringify(leftObservation)).to.equal(JSON.stringify(rightObservation));
      // Sanity: the fixtures really do differ in authority ids/private state.
      expect(leftGame.id).to.not.equal(rightGame.id);
      expect(leftGame.settings.seed).to.not.equal(rightGame.settings.seed);
    });

    it('exposes own private changes only to the owner', () => {
      const left = buildGame({seed: 31, p0Hand: ['CardSwimming']});
      const right = buildGame({seed: 32, p0Hand: ['CardCarnivorous']});
      expect(observe(left.game, 'p0', makeView())).to.not.deep.equal(
        observe(right.game, 'p0', makeView())
      );
      expect(observe(left.game, 'p1', makeView())).to.deep.equal(
        observe(right.game, 'p1', makeView())
      );
    });

    it('shows own hidden traits but omits opponent hidden traits and transient values', () => {
      const withAnglerfish = buildGame({
        seed: 41,
        p0Traits: ['TraitSwimming', {type: 'TraitAnglerfish', value: true}],
      });
      const withoutValue = buildGame({
        seed: 42,
        p0Traits: ['TraitSwimming', {type: 'TraitAnglerfish', value: false}],
      });

      const ownObservation = observe(withAnglerfish.game, 'p0', makeView());
      const anglerfishRow = traitRowOf(ownObservation.animals[0], 'TraitAnglerfish');
      expect(anglerfishRow, 'owner sees hidden Anglerfish').to.not.equal(undefined);
      expect(anglerfishRow.value_kind).to.equal(1);
      expect(anglerfishRow.value_number).to.equal(1);

      const opponentObservation = observe(withAnglerfish.game, 'p1', makeView());
      expect(traitRowOf(opponentObservation.animals[0], 'TraitAnglerfish')).to.equal(undefined);

      expect(observe(withAnglerfish.game, 'p0', makeView())).to.not.deep.equal(
        observe(withoutValue.game, 'p0', makeView())
      );
      expect(observe(withAnglerfish.game, 'p1', makeView())).to.deep.equal(
        observe(withoutValue.game, 'p1', makeView())
      );
    });

    it('derives public carnivorous-plant capabilities without leaking hidden dynamic values', () => {
      const {game} = buildGame();
      const observation = observe(game, 'p1', makeView({actor: 'p1'}));
      const plant = observation.plants[0];
      expect(traitRowOf(plant, PlantTraitHiddenCarnivorous)).to.not.equal(undefined);
      expect(traitRowOf(plant, PlantTraitHiddenIntellect)).to.not.equal(undefined);
      const carnivorous = traitRowOf(plant, PlantTraitHiddenCarnivorous);
      expect(carnivorous.disabled).to.deep.equal([0]);
      expect(carnivorous.cooldowns).to.deep.equal([]);
      expect(carnivorous.value_kind).to.equal(0);
    });
  });

  describe('trait value encoding', () => {
    const buildValueGame = (valueOverrides = {}) => withGameRuntime(
      createRuntime({seed: 77}),
      () => {
        const settings = new SettingsRecord({
          maxPlayers: 2,
          randomPlayers: false,
          addon_plantarium: true,
        });
        let game = GameModel.new({
          id: 'room-values',
          users: List(['p0', 'p1']),
          settings,
        });
        const plant = PlantModel.new(PlantCarnivorous);
        game = game.setIn(['plants', plant.id], plant);

        let animal = AnimalModel.new('p0');
        animal = animal.traitAttach(TraitModel.new('TraitSwimming'));
        const swimming = animal.traits.first();
        animal = animal.traitAttach(TraitModel.new('TraitFatTissue').set('value', true));
        animal = animal.traitAttach(TraitModel.new('TraitCommunication').set('value', {
          sourceType: 'PLANT',
          sourceId: plant.id,
          autoShare: true,
        }));
        animal = animal.traitAttach(TraitModel.new('TraitPlantGrazing').set('value', plant.id));
        animal = animal.traitAttach(TraitModel.new('TraitIntellect').set('value', swimming.id));
        game = game.setIn(['players', 'p0', 'continent', animal.id], animal);

        if (valueOverrides.trait) {
          animal = animal.traitAttach(
            TraitModel.new(valueOverrides.trait).set('value', valueOverrides.value)
          );
          game = game.setIn(['players', 'p0', 'continent', animal.id], animal);
        }

        game = game
          .setIn(['status', 'phase'], PHASE.FEEDING)
          .setIn(['status', 'turn'], 1)
          .setIn(['status', 'round'], 1);
        return game;
      }
    );

    it('encodes booleans, numbers, trait refs, plant refs and feeding records', () => {
      const game = buildValueGame();
      const observation = observe(game, 'p0', makeView());
      const animal = observation.animals[0];

      const fat = traitRowOf(animal, 'TraitFatTissue');
      expect(fat.value_kind).to.equal(1);
      expect(fat.value_number).to.equal(1);

      const communication = traitRowOf(animal, 'TraitCommunication');
      expect(communication.value_kind).to.equal(5);
      expect(communication.source_type).to.equal(
        typeId(catalog, 'feeding_source_types', 'PLANT')
      );
      expect(communication.source_slot).to.equal(0);
      expect(communication.auto_share).to.equal(1);
      expect(communication.value_entity_kind).to.equal(0);

      const grazing = traitRowOf(animal, 'TraitPlantGrazing');
      expect(grazing.value_kind).to.equal(4);
      expect(grazing.value_entity_kind).to.equal(2);
      expect(grazing.value_entity_slot).to.equal(0);

      const intellect = traitRowOf(animal, 'TraitIntellect');
      expect(intellect.value_kind).to.equal(3);
      expect(intellect.value_entity_kind).to.equal(1);
      expect(intellect.value_entity_slot).to.equal(0);
      expect(intellect.value_trait_slot).to.equal(0);

      expectSchemaConformant(observation);
    });

    it('rejects unknown value shapes and drops unresolved raw ids', () => {
      const badObject = buildValueGame({trait: 'TraitFatTissue', value: {nope: true}});
      expect(() => observe(badObject, 'p0', makeView())).to.throw('OBSERVATION_SCHEMA');

      const badId = buildValueGame({trait: 'TraitMimicry', value: 'trait_deadbeef'});
      const observation = observe(badId, 'p0', makeView());
      const row = traitRowOf(observation.animals[0], 'TraitMimicry');
      expect(row.value_kind).to.equal(0);
      expect(JSON.stringify(observation)).to.not.contain('trait_deadbeef');
    });
  });

  describe('purity', () => {
    it('does not consume RNG, ids or clock and is repeatable', () => {
      const runtime = createRuntime({seed: 99});
      const game = withGameRuntime(runtime, () => {
        const settings = new SettingsRecord({
          maxPlayers: 2,
          randomPlayers: false,
          addon_plantarium: true,
        });
        return GameModel.new({id: 'room-pure', users: List(['p0', 'p1']), settings})
          .setIn(['status', 'phase'], PHASE.FEEDING);
      });
      const beforeRuntime = runtime.capture();
      const beforeGame = JSON.stringify(game.toJS());
      const first = observe(game, 'p0', makeView());
      const second = observe(game, 'p0', makeView());
      const afterRuntime = runtime.capture();
      const afterGame = JSON.stringify(game.toJS());

      expect(second).to.deep.equal(first);
      expect(afterRuntime).to.deep.equal(beforeRuntime);
      expect(afterGame).to.equal(beforeGame);
    });
  });
});
