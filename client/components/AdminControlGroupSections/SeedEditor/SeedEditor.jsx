import React, {useEffect, useState} from 'react';
import PropTypes from 'prop-types';

import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import Collapse from '@material-ui/core/Collapse';
import Divider from '@material-ui/core/Divider';
import FormControl from '@material-ui/core/FormControl';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Grid from '@material-ui/core/Grid';
import IconButton from '@material-ui/core/IconButton';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Paper from '@material-ui/core/Paper';
import Select from '@material-ui/core/Select';
import Switch from '@material-ui/core/Switch';
import TextField from '@material-ui/core/TextField';
import Typography from '@material-ui/core/Typography';
import Checkbox from '@material-ui/core/Checkbox';
import ListItemText from '@material-ui/core/ListItemText';
import {makeStyles} from '@material-ui/core/styles';

import DeleteIcon from '@material-ui/icons/Delete';
import AddIcon from '@material-ui/icons/Add';
import CodeIcon from '@material-ui/icons/Code';

import * as cardsData from '~/shared/models/game/evolution/cards/index';
import * as plantTypes from '~/shared/models/game/evolution/plantarium/plantTypes';
import * as traitTypes from '~/shared/models/game/evolution/traitTypes/index';
import {PHASE} from '~/shared/models/game/GameModel';

import {
  buildSeedString,
  defaultSeedString,
  humanizeKey,
  mergePlayersWithRoom,
  parseSeedString,
  getTraitBaseKey,
} from './seedUtils';

const useStyles = makeStyles((theme) => ({
  animalCard: {
    padding: theme.spacing(1),
    marginBottom: theme.spacing(1),
  },
  playerCard: {
    padding: theme.spacing(1.5),
    marginBottom: theme.spacing(1.5),
  },
  editorPaper: {
    padding: theme.spacing(2),
  },
  sectionDivider: {
    margin: theme.spacing(2, 0),
  },
  rawToggleLabel: {
    display: 'flex',
    alignItems: 'center',
  },
  rawToggleIcon: {
    marginRight: theme.spacing(0.5),
  },
}));

const traitOptions = Object.keys(traitTypes).sort().map((key) => ({
  value: key,
  label: humanizeKey(key),
}));

const cardOptions = Object.keys(cardsData)
  .filter((key) => key.startsWith('Card'))
  .sort()
  .map((key) => ({
    value: key,
    label: humanizeKey(key),
  }));

const plantCardOptions = Object.keys(plantTypes)
  .sort()
  .map((key) => ({
    value: key,
    label: humanizeKey(key),
  }));

const phaseOptions = Object.keys(PHASE).map((key) => ({
  value: key.toLowerCase(),
  label: humanizeKey(key),
}));

const seedSettingToggles = [
  'addon_base2',
  'addon_timeToFly',
  'addon_continents',
  'addon_bonus',
  'addon_plantarium',
  'addon_customff',
  'addon_lifecycle',
];

const SeedEditor = ({seed, onChange, roomPlayerCount}) => {
  const [{state, error}, setParsedSeed] = useState(() => parseSeedString(seed || defaultSeedString));
  const [showRaw, setShowRaw] = useState(false);
  const classes = useStyles();

  useEffect(() => {
    setParsedSeed(parseSeedString(seed || defaultSeedString));
  }, [seed]);

  useEffect(() => {
    if (!roomPlayerCount) return;
    if (!state) return;
    if (state.players.length >= roomPlayerCount) return;
    const mergedState = mergePlayersWithRoom(state, roomPlayerCount);
    if (mergedState.players.length !== state.players.length) {
      onChange(buildSeedString(mergedState));
    }
  }, [roomPlayerCount, state, onChange]);

  const emitChange = (nextState) => {
    const newSeed = buildSeedString(nextState);
    onChange(newSeed);
  };

  const getTraitSelections = (traits) => traits.map((trait) => getTraitBaseKey(trait) || trait);

  const handleDeckChange = (index, changes) => {
    const deck = state.deck.map((entry, i) => (i === index ? {...entry, ...changes} : entry));
    emitChange({...state, deck});
  };

  const addDeckEntry = () => {
    const nextCard = cardOptions[0] ? cardOptions[0].value : '';
    emitChange({...state, deck: [...state.deck, {count: 1, card: nextCard}]});
  };

  const removeDeckEntry = (index) => {
    emitChange({...state, deck: state.deck.filter((_, i) => i !== index)});
  };

  const handleDeckPlantsChange = (index, changes) => {
    const deckPlants = state.deckPlants.map((entry, i) => (i === index ? {...entry, ...changes} : entry));
    emitChange({...state, deckPlants});
  };

  const addDeckPlantEntry = () => {
    const nextCard = plantCardOptions[0] ? plantCardOptions[0].value : '';
    emitChange({...state, deckPlants: [...state.deckPlants, {count: 1, card: nextCard}]});
  };

  const removeDeckPlantEntry = (index) => {
    emitChange({...state, deckPlants: state.deckPlants.filter((_, i) => i !== index)});
  };

  const handlePlayerChange = (index, changes) => {
    const players = state.players.map((player, i) => (i === index ? {...player, ...changes} : player));
    emitChange({...state, players});
  };

  const addPlayer = () => {
    emitChange({...state, players: [...state.players, {hand: [], animals: []}]});
  };

  const removePlayer = (index) => {
    emitChange({...state, players: state.players.filter((_, i) => i !== index)});
  };

  const updateHandCard = (playerIndex, cardIndex, changes) => {
    const player = state.players[playerIndex];
    const hand = player.hand.map((entry, i) => (i === cardIndex ? {...entry, ...changes} : entry));
    handlePlayerChange(playerIndex, {hand});
  };

  const addHandCard = (playerIndex) => {
    const player = state.players[playerIndex];
    const hand = [...player.hand, {count: 1, card: cardOptions[0] ? cardOptions[0].value : ''}];
    handlePlayerChange(playerIndex, {hand});
  };

  const removeHandCard = (playerIndex, cardIndex) => {
    const player = state.players[playerIndex];
    const hand = player.hand.filter((_, i) => i !== cardIndex);
    handlePlayerChange(playerIndex, {hand});
  };

  const updateAnimal = (playerIndex, animalIndex, changes) => {
    const player = state.players[playerIndex];
    const animals = player.animals.map((animal, i) => (i === animalIndex ? {...animal, ...changes} : animal));
    handlePlayerChange(playerIndex, {animals});
  };

  const addAnimal = (playerIndex) => {
    const player = state.players[playerIndex];
    const animals = [...player.animals, {food: 0, traits: []}];
    handlePlayerChange(playerIndex, {animals});
  };

  const removeAnimal = (playerIndex, animalIndex) => {
    const player = state.players[playerIndex];
    const animals = player.animals.filter((_, i) => i !== animalIndex);
    handlePlayerChange(playerIndex, {animals});
  };

  const renderDeckList = (list, onEntryChange, onRemove, options = cardOptions, keyPrefix = 'deck') => (
    <Box>
      {list.map((entry, index) => (
        <Grid container spacing={1} alignItems='center' key={`${keyPrefix}-${index}`}>
          <Grid item xs={4}>
            <TextField
              label='Count'
              type='number'
              fullWidth
              value={entry.count}
              onChange={(event) => onEntryChange(index, {count: event.target.value})}
              inputProps={{min: 1}}
            />
          </Grid>
          <Grid item xs={7}>
            <TextField
              select
              label='Card'
              fullWidth
              value={entry.card}
              onChange={(event) => onEntryChange(index, {card: event.target.value})}
            >
              {options.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={1}>
            <IconButton onClick={() => onRemove(index)} aria-label='Remove card'>
              <DeleteIcon fontSize='small'/>
            </IconButton>
          </Grid>
        </Grid>
      ))}
    </Box>
  );

  const renderAnimals = (player, playerIndex) => (
    <Box>
      {player.animals.map((animal, animalIndex) => {
        const traitSelection = getTraitSelections(animal.traits);
        return (
          <Paper key={`animal-${animalIndex}`} className={classes.animalCard} variant='outlined'>
            <Grid container spacing={1} alignItems='center'>
              <Grid item xs={12} sm={4}>
                <TextField
                label='Food tokens'
                type='number'
                fullWidth
                value={animal.food}
                onChange={(event) => updateAnimal(playerIndex, animalIndex, {food: Math.max(0, Number(event.target.value) || 0)})}
                inputProps={{min: 0}}
              />
            </Grid>
              <Grid item xs={12} sm={7}>
                <FormControl fullWidth>
                  <InputLabel id={`traits-${playerIndex}-${animalIndex}`}>Traits</InputLabel>
                  <Select
                    labelId={`traits-${playerIndex}-${animalIndex}`}
                    multiple
                  value={traitSelection}
                  onChange={(event) => {
                    const selectedValue = event.target.value;
                    const selectedTraits = Array.isArray(selectedValue) ? selectedValue : [selectedValue];
                    const existingTraits = new Map();
                    animal.traits.forEach((trait) => {
                      const baseKey = getTraitBaseKey(trait) || trait;
                      if (!existingTraits.has(baseKey)) {
                        existingTraits.set(baseKey, trait);
                      }
                    });
                    const nextTraits = selectedTraits.map((baseKey) => existingTraits.get(baseKey) || baseKey);
                    updateAnimal(playerIndex, animalIndex, {traits: nextTraits});
                  }}
                    renderValue={(selected) => (Array.isArray(selected)
                      ? selected.map((value) => humanizeKey(value)).join(', ')
                      : humanizeKey(selected))}
                  >
                    {traitOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                      <Checkbox checked={traitSelection.includes(option.value)}/>
                      <ListItemText primary={option.label}/>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              </Grid>
              <Grid item xs={12} sm={1}>
                <IconButton onClick={() => removeAnimal(playerIndex, animalIndex)} aria-label='Remove animal'>
                  <DeleteIcon fontSize='small'/>
                </IconButton>
              </Grid>
            </Grid>
          </Paper>
        );
      }))}
      <Button startIcon={<AddIcon/>} onClick={() => addAnimal(playerIndex)} size='small' variant='outlined'>
        Add animal
      </Button>
    </Box>
  );

  const renderHand = (player, playerIndex) => (
    <Box>
      {renderDeckList(
        player.hand,
        (handIndex, changes) => updateHandCard(playerIndex, handIndex, changes),
        (handIndex) => removeHandCard(playerIndex, handIndex),
        cardOptions,
        `hand-${playerIndex}`
      )}
      <Button startIcon={<AddIcon/>} onClick={() => addHandCard(playerIndex)} size='small' variant='outlined'>
        Add card to hand
      </Button>
    </Box>
  );

  const renderPlayers = () => (
    <Box>
      {state.players.map((player, index) => (
        <Paper key={`player-${index}`} className={classes.playerCard} variant='outlined'>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Grid container alignItems='center' justifyContent='space-between'>
                <Typography variant='subtitle1'>Player {index + 1}</Typography>
                <IconButton onClick={() => removePlayer(index)} aria-label='Remove player'>
                  <DeleteIcon fontSize='small'/>
                </IconButton>
              </Grid>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant='subtitle2'>Starting hand</Typography>
              {renderHand(player, index)}
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant='subtitle2'>Continent</Typography>
              {renderAnimals(player, index)}
            </Grid>
          </Grid>
        </Paper>
      ))}
      <Button startIcon={<AddIcon/>} onClick={addPlayer} variant='outlined'>
        Add player
      </Button>
    </Box>
  );

  const renderError = () => (
    error ? (
      <Box mt={1} mb={2}>
        <Typography color='error' variant='body2'>
          Failed to parse existing seed. Editing will use default seed instead.
        </Typography>
      </Box>
    ) : null
  );

  const toggleSeedSetting = (key, checked) => {
    emitChange({
      ...state,
      settings: {
        ...state.settings,
        [key]: checked,
      },
    });
  };

  const updateSeedSettingValue = (key, value) => {
    const nextSettings = {...state.settings};
    if (value === '') {
      delete nextSettings[key];
    } else {
      const currentValue = state.settings[key];
      nextSettings[key] = typeof currentValue === 'number' ? Number(value) || 0 : value;
    }
    emitChange({...state, settings: nextSettings});
  };

  const additionalSettings = Object.keys(state.settings || {})
    .filter((key) => seedSettingToggles.indexOf(key) === -1);

  const handlePhaseChange = (value) => {
    emitChange({...state, phase: value});
  };

  const handleFoodChange = (value) => {
    emitChange({...state, food: Math.max(0, Number(value) || 0)});
  };

  const handleRawChange = (event) => {
    onChange(event.target.value);
  };

  return (
    <Box mt={2}>
      <Paper className={classes.editorPaper} variant='outlined'>
        <Typography variant='h6'>Game seed editor</Typography>
        <Typography variant='body2' color='textSecondary'>
          Configure decks, phase, food and player setups without editing YAML manually.
        </Typography>
        {renderError()}
        <Box mt={2}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                select
                fullWidth
                label='Phase'
                value={state.phase}
                onChange={(event) => handlePhaseChange(event.target.value)}
              >
                {phaseOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                label='Food in watering hole'
                type='number'
                fullWidth
                value={state.food}
                onChange={(event) => handleFoodChange(event.target.value)}
                inputProps={{min: 0}}
              />
            </Grid>
          </Grid>
        </Box>
        <Box mt={3}>
          <Typography variant='subtitle1'>Deck composition</Typography>
          {renderDeckList(state.deck, handleDeckChange, removeDeckEntry)}
          <Box mt={1}>
            <Button startIcon={<AddIcon/>} onClick={addDeckEntry} variant='outlined'>
              Add deck card
            </Button>
          </Box>
        </Box>
        <Box mt={3}>
          <Typography variant='subtitle1'>Seed add-ons</Typography>
          <Grid container>
            {seedSettingToggles.map((key) => (
              <Grid item xs={12} sm={6} md={4} key={key}>
                <FormControlLabel
                  control={(
                    <Checkbox
                      checked={!!state.settings[key]}
                      onChange={(event) => toggleSeedSetting(key, event.target.checked)}
                      color='primary'
                    />
                  )}
                  label={humanizeKey(key)}
                />
              </Grid>
            ))}
          </Grid>
          {additionalSettings.length > 0 ? (
            <Box mt={2}>
              <Typography variant='subtitle2'>Custom seed settings</Typography>
              {additionalSettings.map((key) => (
                <Box key={key} mb={1}>
                  <TextField
                    fullWidth
                    label={humanizeKey(key)}
                    value={state.settings[key] ?? ''}
                    onChange={(event) => updateSeedSettingValue(key, event.target.value)}
                  />
                </Box>
              ))}
            </Box>
          ) : null}
        </Box>
        <Box mt={3}>
          <Grid container alignItems='center' justifyContent='space-between'>
            <Typography variant='subtitle1'>Plant deck (optional)</Typography>
            <Button
              startIcon={<AddIcon/>}
              onClick={addDeckPlantEntry}
              variant='outlined'
              size='small'
              disabled={!state.settings.addon_plantarium}
            >
              Add plant card
            </Button>
          </Grid>
          {!state.settings.addon_plantarium ? (
            <Typography variant='body2' color='textSecondary'>
              Enable the Plantarium addon to include plant cards in the seed.
            </Typography>
          ) : null}
          {renderDeckList(state.deckPlants, handleDeckPlantsChange, removeDeckPlantEntry, plantCardOptions, 'plant')}
        </Box>
        <Box mt={3}>
          <Typography variant='subtitle1'>Players</Typography>
          {renderPlayers()}
        </Box>
        <Divider className={classes.sectionDivider}/>
        <FormControlLabel
          control={<Switch checked={showRaw} onChange={() => setShowRaw(!showRaw)} color='primary'/>}
          label={(<Box className={classes.rawToggleLabel}><CodeIcon className={classes.rawToggleIcon}/>Show raw YAML</Box>)}
        />
        <Collapse in={showRaw}>
          <Box mt={2}>
            <TextField
              label='Raw seed'
              multiline
              rows={6}
              rowsMax={12}
              fullWidth
              value={seed}
              onChange={handleRawChange}
              variant='outlined'
            />
          </Box>
        </Collapse>
      </Paper>
    </Box>
  );
};

SeedEditor.propTypes = {
  seed: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  roomPlayerCount: PropTypes.number,
};

SeedEditor.defaultProps = {
  roomPlayerCount: null,
};

export default SeedEditor;
