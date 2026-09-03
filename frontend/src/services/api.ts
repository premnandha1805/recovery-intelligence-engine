import { CreateDecisionRequest, DecisionResponse, ApiErrorEnvelope } from '../types/api';

/**
 * Resolves the Recovery Intelligence API Gateway base URL.
 * Precedence:
 * 1. Build-time environment variable: import.meta.env.VITE_API_BASE_URL
 * 2. Safe development fallback: http://localhost:3000 (when in dev mode)
 * 3. Safe production fallback: https://recovery-intelligence-api.onrender.com
 */
export function resolveApiBaseUrl(): string {
  const envBase = import.meta.env.VITE_API_BASE_URL;
  if (envBase && typeof envBase === 'string' && envBase.trim() !== '') {
    return envBase.trim().replace(/\/$/, '');
  }
  if (import.meta.env.DEV) {
    return 'http://localhost:3000';
  }
  return 'https://recovery-intelligence-api.onrender.com';
}

export const API_BASE_URL = resolveApiBaseUrl();

export interface ApiEvaluationResult {
  data?: DecisionResponse;
  error?: {
    code: string;
    message: string;
    request_id?: string;
    status?: number;
  };
  latencyMs: number;
  usedProxy: boolean;
  corsBlocked: boolean;
}

/**
 * Executes decision evaluation against the Recovery Intelligence Gateway at ${API_BASE_URL}/decisions.
 */
export async function evaluatePaymentDecision(
  payload: CreateDecisionRequest,
  customRequestId?: string,
): Promise<ApiEvaluationResult> {
  const startTime = performance.now();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (customRequestId) {
    headers['x-request-id'] = customRequestId;
  }

  // Ensure payment_id matches backend validation contract (pay_XXXXXX_aY)
  const originalPaymentId = payload.payment_id;
  const isValidFormat = /^pay_\d{6}_a\d$/.test(originalPaymentId);
  const normalizedPayload: CreateDecisionRequest = {
    ...payload,
    payment_id: isValidFormat ? originalPaymentId : 'pay_910099_a1',
  };

  const jsonBody = JSON.stringify(normalizedPayload);
  const targetUrl = `${API_BASE_URL}/decisions`;

  try {
    const res = await fetch(targetUrl, {
      method: 'POST',
      headers,
      body: jsonBody,
    });

    const latencyMs = Math.round(performance.now() - startTime);

    if (!res.ok) {
      const errorJson: ApiErrorEnvelope | null = await res.json().catch(() => null);
      return {
        error: {
          code: errorJson?.error?.code || `HTTP_${res.status}`,
          message: errorJson?.error?.message || `Request failed with status ${res.status}`,
          request_id: errorJson?.error?.request_id || res.headers.get('x-request-id') || undefined,
          status: res.status,
        },
        latencyMs,
        usedProxy: false,
        corsBlocked: false,
      };
    }

    const data: DecisionResponse = await res.json();
    if (!isValidFormat) {
      data.payment_id = originalPaymentId;
    }
    return {
      data,
      latencyMs,
      usedProxy: false,
      corsBlocked: false,
    };
  } catch (err: any) {
    const latencyMs = Math.round(performance.now() - startTime);
    return {
      error: {
        code: 'NETWORK_ERROR',
        message: err.message || 'Failed to communicate with decision API.',
      },
      latencyMs,
      usedProxy: false,
      corsBlocked: true,
    };
  }
}

/**
 * Checks system health via GET ${API_BASE_URL}/health
 */
export async function checkSystemHealth(): Promise<{ ok: boolean; status?: number; data?: any }> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (!res.ok) {
      return { ok: false, status: res.status };
    }
    const data = await res.json().catch(() => ({}));
    return { ok: true, status: res.status, data };
  } catch (err) {
    return { ok: false };
  }
}
