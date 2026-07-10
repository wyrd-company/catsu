//! Vercel AI Gateway embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, Usage};
use crate::providers::EmbeddingProvider;

const VERCEL_API_URL: &str = "https://ai-gateway.vercel.sh/v1/embeddings";

/// Vercel AI Gateway embeddings provider.
#[derive(Debug)]
pub struct VercelProvider {
    api_key: String,
    http_client: HttpClient,
    base_url: String,
}

impl VercelProvider {
    /// Create a new Vercel AI Gateway provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
            base_url: VERCEL_API_URL.to_string(),
        }
    }

    /// Create a new Vercel AI Gateway provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key =
            std::env::var("AI_GATEWAY_API_KEY").map_err(|_| ClientError::MissingApiKey {
                provider: "vercel".to_string(),
            })?;
        Ok(Self::new(api_key, http_client))
    }

    /// Create a provider that posts to a custom base URL (used by tests).
    #[cfg(test)]
    fn with_base_url(api_key: String, http_client: HttpClient, base_url: String) -> Self {
        Self {
            api_key,
            http_client,
            base_url,
        }
    }
}

/// Vercel AI Gateway API request body (OpenAI-compatible).
#[derive(Debug, Serialize)]
struct VercelEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// Vercel AI Gateway API response.
#[derive(Debug, Deserialize)]
struct VercelEmbeddingResponse {
    data: Vec<VercelEmbedding>,
    model: String,
    usage: VercelUsage,
}

#[derive(Debug, Deserialize)]
struct VercelEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct VercelUsage {
    total_tokens: u64,
}

/// Vercel AI Gateway API error response.
#[derive(Debug, Deserialize)]
struct VercelErrorResponse {
    error: VercelError,
}

#[derive(Debug, Deserialize)]
struct VercelError {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for VercelProvider {
    fn name(&self) -> &'static str {
        "vercel"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending Vercel AI Gateway embedding request"
        );

        // Vercel AI Gateway doesn't support input_type
        let input_type_value = request.input_type;
        if input_type_value.is_some() {
            debug!("Vercel AI Gateway doesn't use input_type parameter, ignoring");
        }

        let input_count = request.inputs.len();
        let body = VercelEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
            dimensions: request.dimensions,
        };

        let body_json = serde_json::to_string(&body)?;
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();
        let url = self.base_url.clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            if let Ok(error_response) = serde_json::from_str::<VercelErrorResponse>(&response_text)
            {
                return Err(ClientError::Api {
                    status,
                    message: error_response.error.message,
                });
            }
            return Err(ClientError::Api {
                status,
                message: response_text,
            });
        }

        let vercel_response: VercelEmbeddingResponse = serde_json::from_str(&response_text)?;

        // Sort embeddings by index
        let mut embeddings: Vec<_> = vercel_response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        let embedding_vectors: Vec<Vec<f32>> =
            embeddings.into_iter().map(|e| e.embedding).collect();

        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = vercel_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: vercel_response.model,
            provider: "vercel".to_string(),
            dimensions,
            input_count,
            input_type: input_type_value,
            latency_ms,
            usage: Usage {
                tokens: total_tokens,
                cost,
            },
        })
    }
}

fn calculate_cost(_model: &str, _tokens: u64) -> Option<f64> {
    // Gateway pricing varies by upstream model and is not fixed in-client.
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::HttpConfig;
    use crate::models::InputType;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn test_name() {
        let provider = VercelProvider::new(
            "test-key".to_string(),
            HttpClient::new(HttpConfig::default()).unwrap(),
        );
        assert_eq!(provider.name(), "vercel");
    }

    #[test]
    fn test_from_env_missing_key() {
        // Ensure the env var is unset for this process check.
        // SAFETY: tests run single-threaded for this case; we restore afterward.
        let previous = std::env::var("AI_GATEWAY_API_KEY").ok();
        std::env::remove_var("AI_GATEWAY_API_KEY");

        let result = VercelProvider::from_env(HttpClient::new(HttpConfig::default()).unwrap());
        assert!(matches!(
            result,
            Err(ClientError::MissingApiKey { provider }) if provider == "vercel"
        ));

        match previous {
            Some(value) => std::env::set_var("AI_GATEWAY_API_KEY", value),
            None => std::env::remove_var("AI_GATEWAY_API_KEY"),
        }
    }

    #[test]
    fn test_calculate_cost_is_none() {
        assert!(calculate_cost("openai/text-embedding-3-small", 1000).is_none());
        assert!(calculate_cost("voyage/voyage-3.5", 500).is_none());
    }

    fn spawn_mock_server(status: u16, body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = body.to_string();

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        });

        format!("http://{addr}/v1/embeddings")
    }

    #[tokio::test]
    async fn test_embed_success_with_mock_server() {
        let body = r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 1},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 0}
            ],
            "model": "openai/text-embedding-3-small",
            "usage": {"prompt_tokens": 8, "total_tokens": 8}
        }"#;
        let url = spawn_mock_server(200, body);

        let provider = VercelProvider::with_base_url(
            "test-key".to_string(),
            HttpClient::new(HttpConfig {
                max_retries: 0,
                ..HttpConfig::default()
            })
            .unwrap(),
            url,
        );

        let response = provider
            .embed(EmbedRequest {
                model: "openai/text-embedding-3-small".to_string(),
                inputs: vec!["hello".to_string(), "world".to_string()],
                input_type: Some(InputType::Query),
                dimensions: Some(3),
                api_key: None,
            })
            .await
            .unwrap();

        assert_eq!(response.provider, "vercel");
        assert_eq!(response.model, "openai/text-embedding-3-small");
        assert_eq!(response.input_count, 2);
        assert_eq!(response.dimensions, 3);
        assert_eq!(response.embeddings.len(), 2);
        // Sorted by index: index 0 then index 1
        assert_eq!(response.embeddings[0], vec![0.4, 0.5, 0.6]);
        assert_eq!(response.embeddings[1], vec![0.1, 0.2, 0.3]);
        assert_eq!(response.usage.tokens, 8);
        assert!(response.usage.cost.is_none());
        assert_eq!(response.input_type, Some(InputType::Query));
    }

    #[tokio::test]
    async fn test_embed_api_error_with_mock_server() {
        let body = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error"}}"#;
        let url = spawn_mock_server(401, body);

        let provider = VercelProvider::with_base_url(
            "bad-key".to_string(),
            HttpClient::new(HttpConfig {
                max_retries: 0,
                ..HttpConfig::default()
            })
            .unwrap(),
            url,
        );

        let err = provider
            .embed(EmbedRequest {
                model: "openai/text-embedding-3-small".to_string(),
                inputs: vec!["hello".to_string()],
                input_type: None,
                dimensions: None,
                api_key: None,
            })
            .await
            .unwrap_err();

        // HttpClient maps non-2xx responses to ClientError::Api before provider parsing.
        match err {
            ClientError::Api { status, message } => {
                assert_eq!(status, 401);
                assert!(message.contains("Invalid API key"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
