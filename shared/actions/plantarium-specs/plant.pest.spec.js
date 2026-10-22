import {makeGameSelectors} from '../../selectors';
import {server$gamePestPlants} from '../game.plantarium';
import {replaceGetRandom} from '../../utils/randomGenerator';

describe('[PLANTARIUM] TraitPest targeting:', function () {
  it('skips the pest map when there are no plants', () => {
    const [{serverStore, ParseGame}] = mockGame(1);
    const gameId = ParseGame(`
settings:
  addon_plantarium: true
phase: feeding
players:
  - continent: $A TraitPest +, $W wait
`);
    const {selectGame} = makeGameSelectors(serverStore.getState, gameId);

    expect(() => serverStore.dispatch(server$gamePestPlants(gameId))).to.not.throw();
    expect(selectGame().plants.size).equal(0);
  });

  it('picks an existing plant when only the last index is rolled', () => {
    const [{serverStore, ParseGame}] = mockGame(1);
    const gameId = ParseGame(`
settings:
  addon_plantarium: true
phase: feeding
plants: PlantEphemeral $eph ++, PlantPerennial $per ++
players:
  - continent: $A TraitPest +, $W wait
`);
    const {selectGame, findPlant} = makeGameSelectors(serverStore.getState, gameId);

    replaceGetRandom((min, max) => max, () => {
      serverStore.dispatch(server$gamePestPlants(gameId));
    });

    expect(findPlant('$per').getFood(), 'last plant is a valid target').equal(1);
    expect(selectGame().plants.size).equal(2);
  });
});
