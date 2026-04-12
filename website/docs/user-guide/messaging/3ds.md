---
sidebar_position: 2
sidebar_label: "Nintendo 3DS"
title: "Nintendo 3DS"
description: "Run Hermes Agent as a native V2 gateway for the Hermes Agent 3DS handheld client"
---

# Nintendo 3DS Setup

Hermes can expose a native **3DS gateway** so a modded Nintendo 3DS can chat with the same agent session model used by other Hermes platforms.

The split is:

- `hermes-agent` — host-side gateway platform
- `hermes-agent-3ds` — handheld client app

The 3DS client speaks a small V2 HTTP API:

- `GET /api/v2/health`
- `GET /api/v2/capabilities`
- `POST /api/v2/messages`
- `GET /api/v2/events`
- `POST /api/v2/interactions/{request_id}/respond`

---

## Prerequisites

- A PC running Hermes Agent
- A modded Nintendo 3DS with the `hermes-agent-3ds` app installed
- The PC and 3DS on the same LAN
- A token configured for the 3DS gateway when exposing it beyond loopback

---

## Configure Hermes

### Option 1: `.env`

Add to `~/.hermes/.env`:

```bash
THREEDS_ENABLED=true
THREEDS_HOST=0.0.0.0
THREEDS_PORT=8787
THREEDS_AUTH_TOKEN=choose-a-long-random-token
THREEDS_DEVICE_ID=old3ds
```

### Option 2: `config.yaml`

Add to `~/.hermes/config.yaml`:

```yaml
platforms:
  3ds:
    enabled: true
    extra:
      host: 0.0.0.0
      port: 8787
      auth_token: choose-a-long-random-token
      device_id: old3ds
```

`device_id` is optional on the gateway side, but giving each handheld a stable ID keeps sessions predictable.

:::warning LAN security
If you bind the gateway to `0.0.0.0` or another network-accessible address, set `THREEDS_AUTH_TOKEN` / `auth_token`.
:::

---

## Start the gateway

```bash
hermes gateway
```

On success you should see a log line like:

```text
[3DS] Listening on 0.0.0.0:8787
```

If you use a host firewall, allow inbound TCP traffic on the chosen port.

---

## Configure the handheld client

In the `hermes-agent-3ds` app on the 3DS, set:

- **Host** — your PC's LAN IP
- **Port** — the 3DS gateway port (for example `8787`)
- **Token** — the same auth token configured in Hermes
- **Device ID** — a stable device name for that handheld

Then:

- press `A` for a health check
- press `B` to send a chat message

---

## 3DS-specific behavior

- **Session identity** is derived from the 3DS device ID and conversation ID
- **Approvals** are delivered through the V2 interaction endpoint, so dangerous terminal commands can still be approved or denied from the handheld
- **Long-poll events** let the 3DS wait for replies without needing WebSockets
- **Allowlist checks are skipped for 3DS** — access control is the device token plus your LAN boundary

---

## Troubleshooting

### Health check fails

1. Confirm Hermes is running with the 3DS platform enabled
2. Confirm the 3DS and PC are on the same network
3. Confirm the host/port match the handheld settings
4. Confirm the firewall allows the selected TCP port

### Unauthorized

The token on the 3DS must exactly match `THREEDS_AUTH_TOKEN` / `platforms.3ds.extra.auth_token`.

### Port already in use

Choose another port with `THREEDS_PORT` or `platforms.3ds.extra.port`.

### Replies never arrive

Check that the client and gateway agree on the same `device_id` and `conversation_id`, and confirm the gateway logs show the incoming request.

---

## Related docs

- [Messaging Gateway overview](/docs/user-guide/messaging)
- [Configuration reference](/docs/user-guide/configuration)
- [Environment variables reference](/docs/reference/environment-variables)
