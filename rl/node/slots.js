/**
 * Local slot assignment shared by the observation projection and the action
 * enumerator.
 *
 * Slots are indices into the ordered lists the policy sees:
 *  - animals: players in fixed seat order (GameModel.players order), then each
 *    player's continent order;
 *  - plants: GameModel.plants order;
 *  - shells: standard area shells order (Immutable Map iteration order).
 *
 * Slot order is meaningful (neoplasm depends on trait/animal order), so it is
 * never sorted by type, id or any other observable value. Slots are valid only
 * within one decision; commands are guarded by decisionId.
 */

export const entitiesOfGame = (game) => {
  const animals = [];
  const animalById = new Map();
  let slot = 0;
  game.players.forEach((player) => {
    let position = 0;
    player.continent.forEach((animal) => {
      const entry = {
        slot
        , id: animal.id
        , ownerId: player.id
        , ownerSeat: player.index
        , position
        , animal
      };
      position += 1;
      animals.push(entry);
      animalById.set(animal.id, entry);
      slot += 1;
    });
  });

  const plants = [];
  const plantById = new Map();
  slot = 0;
  game.plants.forEach((plant) => {
    const entry = {slot, id: plant.id, plant};
    plants.push(entry);
    plantById.set(plant.id, entry);
    slot += 1;
  });

  const shells = [];
  const shellById = new Map();
  slot = 0;
  const area = game.getArea();
  if (area && area.shells) {
    area.shells.forEach((shell) => {
      const entry = {slot, id: shell.id, shell};
      shells.push(entry);
      shellById.set(shell.id, entry);
      slot += 1;
    });
  }

  return {
    animals
    , plants
    , shells
    , animalById
    , plantById
    , shellById
  };
};

export const seatOfPlayer = (game, playerId) => {
  const player = game.getPlayer(playerId);
  return player ? player.index : -1;
};

export const playerIdOfSeat = (game, seat) => {
  const player = game.players.find(player => player.index === seat);
  return player ? player.id : null;
};

export default entitiesOfGame;
