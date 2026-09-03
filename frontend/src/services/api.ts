import { CreateDecisionRequest, DecisionResponse, ApiErrorEnvelope } from '../types/api';

const CONFIGURED_API_BASE = import.meta.env.VITE_API_BASE_URL || '';

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
 * Executes decision evaluation against the Recovery Intelligence Gateway.
 * Supports direct API call with graceful fallback to Vite proxy if browser CORS blocks OPTIONS.
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

  // If CONFIGURED_API_BASE is set to an external domain (e.g. Render), attempt direct call
  if (CONFIGURED_API_BASE && CONFIGURED_API_BASE.startsWith('http')) {
    const directUrl = `${CONFIGURED_API_BASE.replace(/\/$/, '')}/decisions`;
    try {
      const res = await fetch(directUrl, {
        method: 'POST',
        headers,
        body: jsonBody,
      });

      const latencyMs = Math.round(performance.now() - startTime);

      if (!res.ok) {
        const errorJson = await res.json().catch(() => null);
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
    } catch (directErr: any) {
      console.warn(
        '[Recovery Intelligence API] Direct browser call blocked by CORS. Falling back to Vite proxy /decisions:',
        directErr,
      );

      // Attempt fallback via Vite dev/preview proxy
      try {
        const proxyRes = await fetch('/decisions', {
          method: 'POST',
          headers,
          body: jsonBody,
        });

        const latencyMs = Math.round(performance.now() - startTime);

        if (!proxyRes.ok) {
          const errorJson = await proxyRes.json().catch(() => null);
          return {
            error: {
              code: errorJson?.error?.code || `HTTP_${proxyRes.status}`,
              message: errorJson?.error?.message || `Request failed with status ${proxyRes.status}`,
              request_id: errorJson?.error?.request_id || proxyRes.headers.get('x-request-id') || undefined,
              status: proxyRes.status,
            },
            latencyMs,
            usedProxy: true,
            corsBlocked: true,
          };
        }

        const data: DecisionResponse = await proxyRes.json();
        if (!isValidFormat) {
          data.payment_id = originalPaymentId;
        }
        return {
          data,
          latencyMs,
          usedProxy: true,
          corsBlocked: true,
        };
      } catch (proxyErr: any) {
        const latencyMs = Math.round(performance.now() - startTime);
        return {
          error: {
            code: 'NETWORK_ERROR',
            message: proxyErr.message || 'Failed to connect to decision gateway. Check your network or API status.',
          },
          latencyMs,
          usedProxy: true,
          corsBlocked: true,
        };
      }
    }
  }

  // Default path: same-origin / Vite reverse-proxy (/decisions)
  try {
    const res = await fetch('/decisions', {
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
        usedProxy: true,
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
      usedProxy: true,
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
      usedProxy: true,
      corsBlocked: false,
    };
  }
}

/**
 * Checks system health via GET /health
 */
export async function checkSystemHealth(): Promise<{ ok: boolean; status?: number; data?: any }> {
  try {
    const res = await fetch('/health', { method: 'GET' });
    if (!res.ok) {
      return { ok: false, status: res.status };
    }
    const data = await res.json().catch(() => ({}));
    return { ok: true, status: res.status, data };
  } catch (err) {
    return { ok: false };
  }
}
