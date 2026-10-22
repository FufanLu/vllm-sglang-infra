{{- define "vsi.fullname" -}}{{ .Release.Name }}-vsi{{- end -}}
{{- define "vsi.labels" -}}
app.kubernetes.io/name: vllm-sglang-infra
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
{{- define "vsi.modelDeployment" -}}
{{- if .Values.vllm.enabled -}}{{ include "vsi.fullname" . }}-vllm{{- else -}}{{ include "vsi.fullname" . }}-sim{{- end -}}
{{- end -}}
