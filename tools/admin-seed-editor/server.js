const fs = require('fs');
const path = require('path');
const express = require('express');
const parser = require('@babel/parser');
const yaml = require('yaml-js');

const ROOT = path.resolve(__dirname, '..', '..');
const PUBLIC_DIR = path.join(__dirname, 'public');
const CARDS_DIR = path.join(ROOT, 'shared/models/game/evolution/cards');
const I18N_DIR = path.join(ROOT, 'i18n');

const SUPPORTED_LANGUAGES = [
  {code: 'en', file: 'en-en', label: 'English'},
  {code: 'ru', file: 'ru-ru', label: 'Русский'}
];

const toDisplayName = (value = '') => value
  .replace(/^Card/, '')
  .replace(/^Trait/, '')
  .replace(/^PlantTrait/, '')
  .replace(/^Plant/, '')
  .replace(/([a-z])([A-Z])/g, '$1 $2')
  .trim();

const readFile = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');

const parserOptions = {
  sourceType: 'module',
  plugins: [
    'flow',
    'jsx',
    'classProperties',
    'objectRestSpread',
    'optionalChaining',
    'nullishCoalescingOperator',
    'dynamicImport'
  ]
};

const parseModule = (content) => {
  try {
    return parser.parse(content, parserOptions);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Failed to parse source file for admin seed editor library generation:', error.message);
    return null;
  }
};

const extractTraitName = (expression) => {
  if (!expression) return null;
  if (expression.type === 'MemberExpression') {
    if (expression.property) {
      if (expression.property.type === 'Identifier') return expression.property.name;
      if (expression.property.type === 'StringLiteral') return expression.property.value;
    }
    return null;
  }
  if (expression.type === 'Identifier') return expression.name;
  if (expression.type === 'StringLiteral') return expression.value;
  return null;
};

const collectCardNames = () => {
  const files = fs.readdirSync(CARDS_DIR).filter((file) => file.endsWith('.js'));
  const names = new Map();
  files.forEach((file) => {
    const content = fs.readFileSync(path.join(CARDS_DIR, file), 'utf8');
    const ast = parseModule(content);
    if (!ast) return;
    ast.program.body.forEach((node) => {
      if (node.type !== 'ExportNamedDeclaration' || !node.declaration) return;
      if (node.declaration.type !== 'VariableDeclaration') return;
      node.declaration.declarations.forEach((declaration) => {
        if (declaration.id && declaration.id.type === 'Identifier' && declaration.id.name.startsWith('Card')) {
          const traits = [];
          if (declaration.init && declaration.init.type === 'CallExpression') {
            declaration.init.arguments.forEach((argument) => {
              const traitName = extractTraitName(argument);
              if (traitName) traits.push(traitName);
            });
          }
          names.set(declaration.id.name, {id: declaration.id.name, traits});
        }
      });
    });
  });
  names.delete('CardUnknown');
  return Array.from(names.values());
};

const collectTraitNames = () => {
  const content = readFile('shared/models/game/evolution/traitTypes/index.js');
  const ast = parseModule(content);
  if (!ast) return [];
  const result = new Set();
  ast.program.body.forEach((node) => {
    if (node.type !== 'ExportNamedDeclaration' || !node.declaration) return;
    if (node.declaration.type !== 'VariableDeclaration') return;
    node.declaration.declarations.forEach((declaration) => {
      if (!declaration.id || declaration.id.type !== 'Identifier') return;
      if (!declaration.id.name.startsWith('Trait')) return;
      if (declaration.init && declaration.init.type === 'StringLiteral') {
        result.add(declaration.init.value);
      } else {
        result.add(declaration.id.name);
      }
    });
  });
  return Array.from(result);
};

const collectPlantNames = () => {
  const content = readFile('shared/models/game/evolution/plantarium/plantTypes.js');
  const ast = parseModule(content);
  if (!ast) return [];
  const result = new Set();
  ast.program.body.forEach((node) => {
    if (node.type !== 'ExportNamedDeclaration' || !node.declaration) return;
    if (node.declaration.type !== 'VariableDeclaration') return;
    node.declaration.declarations.forEach((declaration) => {
      if (!declaration.id || declaration.id.type !== 'Identifier') return;
      if (!declaration.id.name.startsWith('Plant')) return;
      if (declaration.init && declaration.init.type === 'StringLiteral') {
        result.add(declaration.init.value);
      } else {
        result.add(declaration.id.name);
      }
    });
  });
  return Array.from(result);
};

const collectPhases = () => {
  const content = readFile('shared/models/game/GameModel.js');
  const ast = parseModule(content);
  if (!ast) return [];
  const result = new Set();
  ast.program.body.forEach((node) => {
    if (node.type !== 'ExportNamedDeclaration' || !node.declaration) return;
    if (node.declaration.type !== 'VariableDeclaration') return;
    node.declaration.declarations.forEach((declaration) => {
      if (!declaration.id || declaration.id.type !== 'Identifier' || declaration.id.name !== 'PHASE') return;
      if (!declaration.init || declaration.init.type !== 'ObjectExpression') return;
      declaration.init.properties.forEach((property) => {
        if (property.type !== 'ObjectProperty') return;
        if (property.value && property.value.type === 'StringLiteral') {
          result.add(property.value.value.toLowerCase());
        } else if (property.key) {
          if (property.key.type === 'Identifier') {
            result.add(property.key.name.toLowerCase());
          }
          if (property.key.type === 'StringLiteral') {
            result.add(property.key.value.toLowerCase());
          }
        }
      });
    });
  });
  return Array.from(result);
};

const makeCardLibrary = () => collectCardNames()
  .map(({id, traits}) => {
    const base = id.replace(/^Card/, '');
    const seedName = base.toLowerCase();
    const searchTokens = [id.toLowerCase(), seedName];
    traits.forEach((trait) => searchTokens.push(trait.toLowerCase()));
    return {
      id,
      name: toDisplayName(id),
      seedName,
      category: 'card',
      traits,
      search: searchTokens.join(' ')
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
    category: 'plant',
    search: seedName.toLowerCase()
  }))
  .sort((a, b) => a.name.localeCompare(b.name));

const readLanguageFile = (fileName) => {
  try {
    const filePath = path.join(I18N_DIR, `${fileName}.yml`);
    const content = fs.readFileSync(filePath, 'utf8');
    return yaml.load(content);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn(`Failed to load translations for ${fileName}:`, error.message);
    return null;
  }
};

const normalizeTranslationValue = (value) => {
  if (!value) return null;
  if (typeof value === 'string') return value;
  if (typeof value === 'object') {
    if (typeof value._ === 'string') return value._;
    if (typeof value.name === 'string') return value.name;
    if (typeof value.value === 'string') return value.value;
  }
  return null;
};

const extractTraitTranslations = (doc) => {
  const target = doc && doc.Game && doc.Game.Trait;
  if (!target) return {};
  return Object.entries(target).reduce((acc, [key, value]) => {
    const normalized = normalizeTranslationValue(value);
    if (normalized) acc[key] = normalized;
    return acc;
  }, {});
};

const extractPlantTranslations = (doc) => {
  const target = doc && doc.Game && doc.Game.Plant;
  if (!target) return {};
  return Object.entries(target).reduce((acc, [key, value]) => {
    const normalized = normalizeTranslationValue(value);
    if (normalized) acc[key] = normalized;
    return acc;
  }, {});
};

const buildTranslationBundle = () => {
  const traitLabels = {};
  const plantLabels = {};

  SUPPORTED_LANGUAGES.forEach((language) => {
    const doc = readLanguageFile(language.file);
    traitLabels[language.code] = extractTraitTranslations(doc);
    plantLabels[language.code] = extractPlantTranslations(doc);
  });

  return {
    languages: SUPPORTED_LANGUAGES.map(({code, label}) => ({code, label})),
    traitLabels,
    plantLabels
  };
};

const cardLibrary = makeCardLibrary();
const traitLibrary = makeTraitLibrary();
const plantLibrary = makePlantLibrary();
const phases = collectPhases();
const translationBundle = buildTranslationBundle();

const app = express();
const PORT = process.env.PORT || 4000;

app.use(express.static(PUBLIC_DIR));

app.get('/api/library', (req, res) => {
  res.json({
    cards: cardLibrary,
    traits: traitLibrary,
    plants: plantLibrary,
    phases,
    translations: translationBundle
  });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Admin seed editor running at http://localhost:${PORT}`);
});
