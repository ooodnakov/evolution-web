import fs from 'fs';
import path from 'path';

import {startWorker, runSmoke} from './worker-process.cjs';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const ARTIFACT = path.join(REPO_ROOT, 'dist', 'rl', 'node', 'worker.cjs');
const RUNTIME = 'bun';

const requireArtifact = () => {
  expect(
    fs.existsSync(ARTIFACT),
    `missing build artifact ${ARTIFACT}; run "npm run rl:build"`
  ).to.equal(true);
};

const openWorker = () => {
  requireArtifact();
  return startWorker(RUNTIME, [ARTIFACT], {cwd: REPO_ROOT});
};

const playEpisode = async (worker, seed) => {
  const frames = [];
  await worker.request('hello', {config: {numPlayers: 2}});
  let frame = await worker.request('reset', {seed, scenario: null});
  frames.push(frame);
  while (frame.actor && frames.length < 20000) {
    frame = await worker.request('step', {
      decision_id: frame.decisionId,
      choice_index: 0,
    });
    frames.push(frame);
  }
  const replay = await worker.request('export_replay', {});
  return {frames, replay};
};

describe('rl/node/worker (compiled Bun artifact)', () => {
  it('runs --smoke to a natural terminal', async function () {
    this.timeout(30000);
    requireArtifact();
    const {code, stdout, stderr} = await runSmoke(
      RUNTIME,
      [ARTIFACT, '--smoke'],
      {cwd: REPO_ROOT}
    );
    const lines = stdout.trim().split('\n');
    expect(code, stderr).to.equal(0);
    expect(lines).to.have.length(1);
    const report = JSON.parse(lines[0]);
    expect(report.ok).to.equal(true);
    expect(report.terminated).to.equal(true);
    expect(report.truncated).to.equal(false);
    expect(report.decisions).to.be.greaterThan(0);
  });

  it('speaks clean JSONL to a natural terminal', async function () {
    this.timeout(30000);
    const schema = JSON.parse(
      fs.readFileSync(path.join(REPO_ROOT, 'rl', 'schema.json'), 'utf8')
    );
    const worker = openWorker();
    try {
      const hello = await worker.request('hello', {config: {numPlayers: 2}});
      expect(hello.protocol_version).to.equal(schema.protocol_version);
      const {frames, replay} = await playEpisode(worker, 7);
      const last = frames[frames.length - 1];
      expect(last.actor).to.equal(null);
      expect(Object.values(last.terminated).every(Boolean)).to.equal(true);
      expect(replay.checkpoints.length).to.be.greaterThan(0);
    } finally {
      await worker.close();
    }
  });

  it('replays identically in two fresh processes', async function () {
    this.timeout(30000);
    const first = openWorker();
    const second = openWorker();
    try {
      const a = await playEpisode(first, 123);
      const b = await playEpisode(second, 123);
      expect(JSON.stringify(b.frames)).to.equal(JSON.stringify(a.frames));
      expect(JSON.stringify(b.replay)).to.equal(JSON.stringify(a.replay));
    } finally {
      // Both children must be reaped even if the first cleanup fails.
      const settled = await Promise.allSettled([first.close(), second.close()]);
      const rejected = settled.find((result) => result.status === 'rejected');
      if (rejected) throw rejected.reason;
    }
  });
});
