#!/usr/bin/env node
'use strict';

/*
 * Ahead-of-time Babel build for the headless RL engine.
 *
 * Writes the mirrored CommonJS tree that Bun (and Node) execute:
 *
 *   dist/rl/node/worker.cjs   JSONL launcher (plain CommonJS, no Babel at runtime)
 *   dist/rl/node/*.js         compiled engine
 *   dist/rl/shared/**         compiled rules core
 *   dist/rl/server/**         compiled reducer/store modules used by the engine
 *   dist/rl/schema.json       protocol contract, read by the worker and by Python
 *
 * `babel-plugin-root-import` is pinned to dist/rl, so `~/shared/...` becomes a
 * relative require that stays inside the artifact. ES5 output is required: it
 * keeps `class X extends Record(...)` constructors callable, which Immutable
 * 3.8.2 `reify()` relies on and native class semantics break.
 *
 * Deterministic: the output directory is replaced wholesale, files are walked
 * in sorted order and Babel writes no source maps or timestamps. Unrelated
 * webpack output under dist/ is never touched.
 *
 * Run: npm run rl:build
 */

const fs = require('fs');
const path = require('path');
const babel = require('@babel/core');

const REPO_ROOT = path.resolve(__dirname, '..');
const OUT = path.join(REPO_ROOT, 'dist', 'rl');
const COPY_EXTENSIONS = new Set(['.js', '.cjs', '.json']);

const SOURCES = [
  {from: path.join(REPO_ROOT, 'shared'), to: path.join(OUT, 'shared')},
  {from: path.join(REPO_ROOT, 'server'), to: path.join(OUT, 'server')},
  {from: path.join(REPO_ROOT, 'rl', 'node'), to: path.join(OUT, 'node'), skipDirs: ['test']},
];

const babelrcPath = path.join(REPO_ROOT, '.babelrc');
if (!fs.existsSync(babelrcPath)) {
  throw new Error('missing ' + babelrcPath + '; the RL build uses the repository Babel config');
}
const babelrc = JSON.parse(fs.readFileSync(babelrcPath, 'utf8'));

const plugins = (babelrc.plugins || []).map((plugin) => {
  const name = typeof plugin === 'string' ? plugin : plugin[0];
  // The copied tree has the same shape as the source tree, but its root is
  // dist/rl: point the root-import rewrite at the artifact, not at the repo.
  return name === 'babel-plugin-root-import'
    ? ['babel-plugin-root-import', {root: OUT}]
    : plugin;
});

// Locale-independent ordering: the build output must not depend on ICU state.
const byName = (a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0);

function copyTree(from, to, skipDirs = []) {
  let copied = 0;
  fs.mkdirSync(to, {recursive: true});
  for (const entry of fs.readdirSync(from, {withFileTypes: true}).sort(byName)) {
    const src = path.join(from, entry.name);
    const dest = path.join(to, entry.name);
    if (entry.isDirectory()) {
      if (skipDirs.includes(entry.name)) continue;
      copied += copyTree(src, dest);
      continue;
    }
    if (!COPY_EXTENSIONS.has(path.extname(entry.name))) continue;
    if (entry.name.endsWith('.spec.js')) continue;
    fs.copyFileSync(src, dest);
    copied += 1;
  }
  return copied;
}

function walk(dir, acc) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true}).sort(byName)) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(file, acc);
    else acc.push(file);
  }
  return acc;
}

// Only dist/rl belongs to this build; dist/server and dist/client are webpack's.
fs.rmSync(OUT, {recursive: true, force: true});

let copied = 0;
for (const source of SOURCES) {
  copied += copyTree(source.from, source.to, source.skipDirs || []);
}
fs.copyFileSync(path.join(REPO_ROOT, 'rl', 'schema.json'), path.join(OUT, 'schema.json'));
copied += 1;

let compiled = 0;
for (const file of walk(OUT, [])) {
  if (!file.endsWith('.js')) continue;
  const result = babel.transformFileSync(file, {
    cwd: OUT,
    filename: file,
    babelrc: false,
    configFile: false,
    presets: babelrc.presets || [],
    plugins,
  });
  fs.writeFileSync(file, result.code);
  compiled += 1;
}

// Progress goes to stderr so `npm run rl:smoke` keeps a single JSON line on stdout.
console.error('rl build: compiled ' + compiled + ' modules, copied ' + copied + ' files -> ' + path.relative(REPO_ROOT, OUT));
