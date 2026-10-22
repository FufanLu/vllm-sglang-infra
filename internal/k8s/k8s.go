// Package k8s is the small slice of the Kubernetes API the gateway uses:
// listing model-server pods by label, and scaling their Deployment. It talks
// REST with the pod's service-account credentials, so the gateway needs no
// client library and no kubeconfig inside the cluster.
package k8s

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const saDir = "/var/run/secrets/kubernetes.io/serviceaccount"

type Client struct {
	Base      string // https://host:port
	Token     string
	Namespace string
	HTTP      *http.Client
}

// InCluster builds a client from the pod's service account.
func InCluster() (*Client, error) {
	host, port := os.Getenv("KUBERNETES_SERVICE_HOST"), os.Getenv("KUBERNETES_SERVICE_PORT")
	if host == "" || port == "" {
		return nil, fmt.Errorf("not running in a Kubernetes pod (KUBERNETES_SERVICE_HOST unset)")
	}
	token, err := os.ReadFile(saDir + "/token")
	if err != nil {
		return nil, err
	}
	ns, err := os.ReadFile(saDir + "/namespace")
	if err != nil {
		return nil, err
	}
	ca, err := os.ReadFile(saDir + "/ca.crt")
	if err != nil {
		return nil, err
	}
	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(ca)
	return &Client{
		Base:      "https://" + net.JoinHostPort(host, port),
		Token:     strings.TrimSpace(string(token)),
		Namespace: strings.TrimSpace(string(ns)),
		HTTP: &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{
			TLSClientConfig: &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12},
		}},
	}, nil
}

func (c *Client) do(ctx context.Context, method, path, contentType string, body []byte, out any) error {
	req, err := http.NewRequestWithContext(ctx, method, c.Base+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.Token)
	req.Header.Set("Accept", "application/json")
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("%s %s: %s: %s", method, path, resp.Status, strings.TrimSpace(string(msg)))
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

type Pod struct {
	Name        string
	IP          string
	Ready       bool
	Running     bool
	Deleting    bool
	Labels      map[string]string
	Annotations map[string]string
}

// Pods lists the namespace's pods matching a label selector such as
// "app=vllm,model=qwen3-8b".
func (c *Client) Pods(ctx context.Context, selector string) ([]Pod, error) {
	var list struct {
		Items []struct {
			Metadata struct {
				Name              string            `json:"name"`
				Labels            map[string]string `json:"labels"`
				Annotations       map[string]string `json:"annotations"`
				DeletionTimestamp *string           `json:"deletionTimestamp"`
			} `json:"metadata"`
			Status struct {
				Phase      string `json:"phase"`
				PodIP      string `json:"podIP"`
				Conditions []struct {
					Type   string `json:"type"`
					Status string `json:"status"`
				} `json:"conditions"`
			} `json:"status"`
		} `json:"items"`
	}
	path := "/api/v1/namespaces/" + url.PathEscape(c.Namespace) + "/pods?labelSelector=" + url.QueryEscape(selector)
	if err := c.do(ctx, http.MethodGet, path, "", nil, &list); err != nil {
		return nil, err
	}
	pods := make([]Pod, 0, len(list.Items))
	for _, it := range list.Items {
		p := Pod{Name: it.Metadata.Name, IP: it.Status.PodIP, Labels: it.Metadata.Labels,
			Annotations: it.Metadata.Annotations, Running: it.Status.Phase == "Running",
			Deleting: it.Metadata.DeletionTimestamp != nil}
		for _, cond := range it.Status.Conditions {
			if cond.Type == "Ready" && cond.Status == "True" {
				p.Ready = true
			}
		}
		pods = append(pods, p)
	}
	return pods, nil
}

// Scale reads a Deployment's desired replica count.
func (c *Client) Scale(ctx context.Context, deployment string) (int, error) {
	var s struct {
		Spec struct {
			Replicas int `json:"replicas"`
		} `json:"spec"`
	}
	err := c.do(ctx, http.MethodGet, c.scalePath(deployment), "", nil, &s)
	return s.Spec.Replicas, err
}

// SetScale sets a Deployment's desired replica count.
func (c *Client) SetScale(ctx context.Context, deployment string, replicas int) error {
	patch := fmt.Appendf(nil, `{"spec":{"replicas":%d}}`, replicas)
	return c.do(ctx, http.MethodPatch, c.scalePath(deployment), "application/merge-patch+json", patch, nil)
}

func (c *Client) scalePath(deployment string) string {
	return "/apis/apps/v1/namespaces/" + url.PathEscape(c.Namespace) + "/deployments/" + url.PathEscape(deployment) + "/scale"
}
