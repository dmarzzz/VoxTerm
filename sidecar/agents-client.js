// Wrapper around @fileverse/agents that gives us:
//   - one-time agent setup keyed by NAMESPACE
//   - createOrUpdate(session_id, markdown, store)
//       * fresh session_id -> agent.create + record (session_id -> file_id) in store
//       * known session_id -> agent.update on the existing file_id
//
// The SDK is interface-only here; tests inject a fake to avoid touching
// real Pinata / Pimlico / Gnosis. Real wiring matches the SDK's documented
// usage:
//   https://www.npmjs.com/package/@fileverse/agents
//   https://github.com/fileverse/agents
//
// Returned shape per call: { file_id, tx_hash, block_number }.
// tx_hash / block_number may be null if the SDK didn't surface them.

import { privateKeyToAccount } from 'viem/accounts';

let _agent = null;

/**
 * Build (and memoize) the Agent for this process. Called lazily so that
 * an unconfigured deploy fails on the first /v1/transcripts request rather
 * than at import time, which keeps `npm test` running without env vars.
 */
export async function getAgent({ env = process.env, agentFactory } = {}) {
  if (_agent) return _agent;

  // Only enforce required env vars when there's no test-injected factory;
  // tests legitimately don't have a Pinata JWT.
  if (!agentFactory) {
    const required = ['PRIVATE_KEY', 'PIMLICO_API_KEY', 'PINATA_JWT', 'NAMESPACE'];
    const missing = required.filter((k) => !env[k]);
    if (missing.length) {
      throw new Error(
        `sidecar misconfigured: missing env vars ${missing.join(', ')} (see .env.example)`
      );
    }
  }

  // The factory hook lets tests inject a fake without touching the SDK.
  // Real factory dynamically imports so we don't pay the SDK's startup cost
  // (or fail) when running unit tests.
  const factory =
    agentFactory ??
    (async () => {
      const [{ Agent }, { PinataStorageProvider }] = await Promise.all([
        import('@fileverse/agents'),
        import('@fileverse/agents/storage'),
      ]);
      const agent = new Agent({
        chain: 'gnosis',
        viemAccount: privateKeyToAccount(env.PRIVATE_KEY),
        pimlicoAPIKey: env.PIMLICO_API_KEY,
        storageProvider: new PinataStorageProvider({
          jwt: env.PINATA_JWT,
          gateway: env.PINATA_GATEWAY || 'gateway.pinata.cloud',
        }),
      });
      await agent.setupStorage(env.NAMESPACE);
      return agent;
    });

  _agent = await factory();
  return _agent;
}

/**
 * Idempotent on session_id: creates a new on-chain entry the first time,
 * updates the existing file (Agent SDK records a new revision) on retries.
 */
export async function createOrUpdate({ session_id, markdown, store, env, agentFactory }) {
  const agent = await getAgent({ env, agentFactory });
  const existing = store.get(session_id);

  if (existing && existing.file_id) {
    const result = await agent.update(existing.file_id, markdown);
    return _normalize(result, existing.file_id);
  } else {
    const result = await agent.create(markdown);
    return _normalize(result);
  }
}

function _normalize(result, fallbackFileId = null) {
  // The SDK's return shape varies across versions; normalize to the three
  // fields we actually use. Anything missing comes back as null so the API
  // response is predictable.
  if (!result) {
    return { file_id: fallbackFileId, tx_hash: null, block_number: null };
  }
  return {
    file_id:
      result.fileId ?? result.file_id ?? result.id ?? fallbackFileId ?? null,
    tx_hash: result.txHash ?? result.tx_hash ?? result.transactionHash ?? null,
    block_number:
      result.blockNumber ?? result.block_number ?? result.block ?? null,
  };
}

/** For tests only — drop the cached agent so a fresh factory runs. */
export function _resetForTests() {
  _agent = null;
}
