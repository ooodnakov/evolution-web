/**
 * Mocha setup for the headless RL specs.
 *
 * Deliberately minimal: Chai assertions and nothing else. No `globals.js`,
 * no `shared/test-helper.js`, no JWT/Mongo environment, so the RL node tests
 * fail if a headless module pulls a web-only dependency.
 */
const chai = require('chai');

global.chai = chai;
global.expect = chai.expect;
