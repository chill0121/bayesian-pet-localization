# Hardware Configuration

Settings changed from factory defaults for the Bayesian Pet Localization system.

## Beacon — BlueCharm BC021 Pro

Configured via BlueCharm Toolbox app.

| Setting | Default | Changed To |
|---|---|---|
| SLOT 0 Protocol | iBeacon | — (no change) |
| SLOT 0 Advertising Interval | 200 ms | **211.25 ms** |
| SLOT 0 TX Power | 0 dBm | **+4 dBm** |
| SLOT 0 Measured Power | -59 dBm | **-55 dBm** |
| SLOT 1 Protocol | OFF | **TLM (Eddystone)** |
| SLOT 1 Advertising Interval | — | **10000 ms** |
| Battery | CR2032 | — |
| iBeacon UUID | `426c7565-4368-6172-6d42-6561636f6e73` | — |
| iBeacon Major | `3838` | — |
| iBeacon Minor | `4949` | — |

## Anchors — ESP32-S3 (ESPresense v4.0.6)

9 anchors across 3 floors. All share identical settings except Room Name and IP.

### Network (per anchor)

| Room Name | IP Address | WiFi SSID | WiFi Password |
|---|---|---|---|
| 1F_Office | 192.168.0.188 | `IoT` | *(set in device)* |
| 1F_Hallway | 192.168.0.249 | `IoT` | *(set in device)* |
| 2F_Kitchen_NE | 192.168.0.196 | `IoT` | *(set in device)* |
| 2F_Living_Center | 192.168.0.35 | `IoT` | *(set in device)* |
| 2F_Living_SW | 192.168.0.43 | `IoT` | *(set in device)* |
| 2F_Living_SE | 192.168.0.71 | `IoT` | *(set in device)* |
| 3F_Hallway | 192.168.0.123 | `IoT` | *(set in device)* |
| 3F_Master_Bed | 192.168.0.209 | `IoT` | *(set in device)* |
| 3F_Office | 192.168.0.180 | `IoT` | *(set in device)* |

### MQTT

| Setting | Value |
|---|---|
| Server | `192.168.0.104` |
| Port | `1883` |

### Filtering (changed from defaults)

| Setting | Default | Changed To |
|---|---|---|
| Maximum distance | 16 m | **40 m** |
| Skip reporting | 5000 ms | **2000 ms** |
| Include only IDs | *(empty)* | **`iBeacon:426c7565-4368-6172-6d42-6561636f6e73`** |

### Calibration (unchanged from defaults)

| Setting | Value | Notes |
|---|---|---|
| tx_ref_rssi | -59 | Not used for iBeacon devices |
| rx_adj_rssi | 20 | Per-receiver offset |
| absorption | 2.70 | Environmental factor |

### Hardware User Manual Links
https://docs.m5stack.com/en/core/AtomS3%20Lite

https://bluecharmbeacons.com/bc021-pro-ibeacon-deep-dive-into-the-configuration-screens/#TX%20Power