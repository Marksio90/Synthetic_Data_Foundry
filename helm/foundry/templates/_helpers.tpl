{{/*
Expand the name of the chart.
*/}}
{{- define "foundry.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "foundry.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "foundry.labels" -}}
helm.sh/chart: {{ include "foundry.name" . }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "foundry.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "foundry.selectorLabels" -}}
app.kubernetes.io/name: {{ include "foundry.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
