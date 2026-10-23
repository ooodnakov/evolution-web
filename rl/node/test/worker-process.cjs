'use strict';

/*
 * Bounded lifecycle for the compiled-worker subprocess spec.
 *
 * A hung child must never keep Mocha's event loop alive: every wait has a
 * deadline, a timed-out request rejects pending work, kills the child and
 * awaits its exit, and `close()` still reaps the child when the protocol
 * handshake fails.
 */

const {spawn} = require('child_process');

const REQUEST_TIMEOUT_MS = 20000;
const EXIT_TIMEOUT_MS = 5000;

const bounded = (promise, ms, message) =>
  Promise.race([
    promise,
    new Promise((ignore, reject) => {
      const timer = setTimeout(() => reject(new Error(message)), ms);
      if (typeof timer.unref === 'function') timer.unref();
    }),
  ]);

const waitExit = (exitPromise, isExited, ms, label) =>
  isExited()
    ? Promise.resolve()
    : bounded(exitPromise, ms, `${label} did not exit within ${ms}ms`);

function startWorker(runtime, argv, options = {}) {
  const requestTimeoutMs = options.requestTimeoutMs || REQUEST_TIMEOUT_MS;
  const exitTimeoutMs = options.exitTimeoutMs || EXIT_TIMEOUT_MS;
  const child = spawn(runtime, argv, {
    cwd: options.cwd,
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  const pending = new Map();
  const stderr = [];
  let nextId = 1;
  let buffer = '';
  let exited = false;
  let exitInfo = null;
  let failure = null;
  let resolveExit;
  const exitPromise = new Promise((resolve) => {
    resolveExit = resolve;
  });

  const rejectPending = (error) => {
    for (const entry of pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(error);
    }
    pending.clear();
  };

  const killAndWait = async (error) => {
    if (error && !failure) failure = error;
    rejectPending(error || failure || new Error('worker stopped'));
    if (!exited) child.kill('SIGKILL');
    return waitExit(exitPromise, () => exited, exitTimeoutMs, 'worker');
  };

  const failStream = (error) => {
    if (!failure) failure = error;
    rejectPending(failure);
    if (!exited) child.kill('SIGKILL');
  };

  child.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    let index = buffer.indexOf('\n');
    while (index >= 0) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      index = buffer.indexOf('\n');
      if (!line) continue;
      // Any non-JSON stdout byte is a stream-contract violation and is fatal.
      try {
        const response = JSON.parse(line);
        const entry = pending.get(response.id);
        if (!entry) throw new Error(`unexpected response id ${response.id}`);
        pending.delete(response.id);
        clearTimeout(entry.timer);
        if (response.ok) entry.resolve(response.result);
        else {
          entry.reject(
            Object.assign(new Error(response.error.message), {code: response.error.code})
          );
        }
      } catch (error) {
        failStream(error);
      }
    }
  });
  child.stderr.on('data', (chunk) => stderr.push(chunk.toString()));
  child.on('error', (error) => failStream(error));
  child.on('close', (code, signal) => {
    exited = true;
    exitInfo = {code, signal};
    rejectPending(failure || new Error(`worker exited with code ${code}: ${stderr.join('')}`));
    resolveExit(exitInfo);
  });

  const request = (op, payload = {}) => {
    if (failure) return Promise.reject(failure);
    if (exited) return Promise.reject(new Error('worker already exited'));
    return new Promise((resolve, reject) => {
      const id = nextId++;
      const timer = setTimeout(() => {
        pending.delete(id);
        const error = new Error(`no response to ${op} within ${requestTimeoutMs}ms`);
        error.code = 'WORKER_TIMEOUT';
        reject(error);
        // Reap the hung child; close() will observe the failure and also wait.
        void killAndWait(error).catch(() => {});
      }, requestTimeoutMs);
      if (typeof timer.unref === 'function') timer.unref();
      pending.set(id, {
        timer,
        resolve,
        reject,
      });
      child.stdin.write(JSON.stringify({id, op, payload}) + '\n');
    });
  };

  const close = async () => {
    if (exited) return;
    if (!failure) {
      try {
        await request('close', {});
      } catch (error) {
        failure = failure || error;
      }
    }
    if (!failure) {
      try {
        await waitExit(exitPromise, () => exited, exitTimeoutMs, 'worker after close');
        return;
      } catch (error) {
        failure = error;
      }
    }
    await killAndWait(failure);
  };

  return {
    pid: child.pid,
    exited: () => exited,
    request,
    close,
    stderrText: () => stderr.join(''),
  };
}

function runSmoke(runtime, argv, options = {}) {
  const timeoutMs = options.timeoutMs || REQUEST_TIMEOUT_MS;
  const exitTimeoutMs = options.exitTimeoutMs || EXIT_TIMEOUT_MS;
  const child = spawn(runtime, argv, {
    cwd: options.cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stdout = '';
  let stderr = '';
  let exited = false;
  let settled = false;
  let resolveExit;
  const exitPromise = new Promise((resolve) => {
    resolveExit = resolve;
  });
  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString();
  });
  child.stderr.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  return new Promise((resolve, reject) => {
    const timer = setTimeout(async () => {
      if (settled) return;
      settled = true;
      if (!exited) child.kill('SIGKILL');
      try {
        await waitExit(exitPromise, () => exited, exitTimeoutMs, 'smoke worker');
      } catch (error) {
        reject(error);
        return;
      }
      reject(new Error(`smoke worker did not exit within ${timeoutMs}ms`));
    }, timeoutMs);
    if (typeof timer.unref === 'function') timer.unref();
    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (!exited) child.kill('SIGKILL');
      reject(error);
    });
    child.on('close', (code, signal) => {
      exited = true;
      resolveExit({code, signal});
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({code, signal, stdout, stderr});
    });
  });
}

module.exports = {startWorker, runSmoke};
