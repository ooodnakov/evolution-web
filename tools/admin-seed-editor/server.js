const fs = require('fs');
const path = require('path');
const express = require('express');
const bodyParser = require('body-parser');

const ROOT = path.resolve(__dirname, '..', '..');
const PUBLIC_DIR = path.join(__dirname, 'public');
const CARDS_DIR = path.join(ROOT, 'shared/models/game/evolution/cards');

const toDisplayName = (value = '') => value
  .replace(/^Card/, '')
  .replace(/^Trait/, '')
  .replace(/^PlantTrait/, '')
  .replace(/^Plant/, '')
  .replace(/([a-z])([A-Z])/g, '$1 $2')
  .trim();

const readFile = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');

const collectCardNames = () => {
  const files = fs.readdirSync(CARDS_DIR).filter((file) => file.endsWith('.js'));
  const names = new Set();
  files.forEach((file) => {
    const content = fs.readFileSync(path.join(CARDS_DIR, file), 'utf8');
    const regex = /export const (Card[\w]+)/g;
    let match;
    while ((match = regex.exec(content)) !== null) {
      names.add(match[1]);
    }
  });
  names.delete('CardUnknown');
  return Array.from(names);
};

const collectTraitNames = () => {
  const content = readFile('shared/models/game/evolution/traitTypes/index.js');
  const regex = /export const (Trait[\w]+)\s*=\s*'([^']+)'/g;
  const result = [];
  let match;
  while ((match = regex.exec(content)) !== null) {
    result.push(match[2]);
  }
  return Array.from(new Set(result));
};

const collectPlantNames = () => {
  const content = readFile('shared/models/game/evolution/plantarium/plantTypes.js');
  const regex = /export const (Plant[\w]+)\s*=\s*'([^']+)'/g;
  const result = [];
  let match;
  while ((match = regex.exec(content)) !== null) {
    result.push(match[2]);
  }
  return Array.from(new Set(result));
};

const collectPhases = () => {
  const content = readFile('shared/models/game/GameModel.js');
  const blockMatch = content.match(/export const PHASE = \{([\s\S]*?)\};/);
  if (!blockMatch) return [];
  const block = blockMatch[1];
  const regex = /([A-Z_]+)\s*:/g;
  const phases = [];
  let match;
  while ((match = regex.exec(block)) !== null) {
    phases.push(match[1].toLowerCase());
  }
  return Array.from(new Set(phases));
};

const makeCardLibrary = () => collectCardNames()
  .map((name) => {
    const base = name.replace(/^Card/, '');
    const seedName = base.toLowerCase();
    return {
      id: name,
      name: toDisplayName(name),
      seedName,
      search: `${name.toLowerCase()} ${seedName}`
    };
  })
  .sort((a, b) => a.name.localeCompare(b.name));

const makeTraitLibrary = () => collectTraitNames()
  .map((seedName) => ({
    id: seedName,
    name: toDisplayName(seedName),
    seedName,
    search: seedName.toLowerCase()
  }))
  .sort((a, b) => a.name.localeCompare(b.name));

const makePlantLibrary = () => collectPlantNames()
  .map((seedName) => ({
    id: seedName,
    name: toDisplayName(seedName),
    seedName,
    search: seedName.toLowerCase()
  }))
  .sort((a, b) => a.name.localeCompare(b.name));

const cardLibrary = makeCardLibrary();
const traitLibrary = makeTraitLibrary();
const plantLibrary = makePlantLibrary();
const phases = collectPhases();

const app = express();
const PORT = process.env.PORT || 4000;

app.use(bodyParser.json({limit: '1mb'}));
app.use(express.static(PUBLIC_DIR));

app.get('/api/library', (req, res) => {
  res.json({
    cards: cardLibrary,
    traits: traitLibrary,
    plants: plantLibrary,
    phases
  });
});

app.post('/api/library/cards/resolve', (req, res) => {
  const {tokens = []} = req.body || {};
  const resolved = tokens.map((token) => {
    if (typeof token !== 'string') return null;
    const normalized = token.trim().toLowerCase();
    const match = cardLibrary.find((card) => card.seedName === normalized
      || card.search.includes(normalized));
    return match ? match.seedName : normalized;
  }).filter(Boolean);
  res.json({tokens: resolved});
});

app.get('*', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Admin seed editor running at http://localhost:${PORT}`);
});
