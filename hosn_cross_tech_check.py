#!/usr/bin/env python3
"""Read-only HOSN cross-technology capability check.

This script does not install packages, load kernel modules, start Mininet,
create interfaces, or change the network.  It only inspects the Ubuntu VM so
we can decide what second access technology can be implemented honestly.

Run from the Hosn repository without sudo:
    python3 hosn_cross_tech_check.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def command(argv: list[str]) -> dict:
    """Run a read-only inspection command and preserve its result."""
    try:
        completed = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=8,
            check=False,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "output": completed.stdout.strip()[:4000],
        }
    except Exception as exc:
        return {
            "argv": argv,
            "returncode": None,
            "output": type(exc).__name__ + ": " + str(exc),
        }


def module_probe(name: str) -> dict:
    """Inspect a kernel module without loading it."""
    loaded = False
    try:
        loaded = any(
            line.split()[0] == name
            for line in Path("/proc/modules").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError:
        pass
    result = {
        "name": name,
        "currently_loaded": loaded,
        "modinfo": command(["modinfo", name]) if shutil.which("modinfo") else None,
        # --dry-run and --show-depends must never insert the module.
        "modprobe_dry_run": command(["modprobe", "--dry-run", "--show-depends", name])
        if shutil.which("modprobe") else None,
    }
    return result


def main() -> int:
    report: dict = {
        "checker": "hosn-cross-tech-read-only-v1",
        "changes_made": False,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "commands": {},
        "kernel_modules": {},
        "mininet_wifi": {},
    }

    for name in ("ip", "iw", "iwpan", "tc", "modinfo", "modprobe", "mmcli"):
        report["commands"][name] = shutil.which(name)

    for name in ("mac80211_hwsim", "mac802154_hwsim", "wwan_hwsim"):
        report["kernel_modules"][name] = module_probe(name)

    try:
        mn_wifi = importlib.import_module("mn_wifi")
        net_module = importlib.import_module("mn_wifi.net")
        cls = net_module.Mininet_wifi
        package_dir = Path(mn_wifi.__file__).resolve().parent
        repo_dir = package_dir.parent
        report["mininet_wifi"] = {
            "import_ok": True,
            "version": str(getattr(net_module, "VERSION", "unknown")),
            "package_dir": str(package_dir),
            "class_mro": [c.__name__ for c in cls.__mro__],
            "has_addStation": hasattr(cls, "addStation"),
            "has_addSensor_6lowpan": hasattr(cls, "addSensor"),
            "has_addModem_wwan": hasattr(cls, "addModem"),
            "has_addBTDevice": hasattr(cls, "addBTDevice"),
            "configureNodes_signature": str(inspect.signature(cls.configureNodes)),
            "examples": {
                name: (repo_dir / "examples" / name).is_file()
                for name in (
                    "6LoWPan.py",
                    "wmediumd_interference_lowpan.py",
                    "wwan.py",
                    "ieee80211p.py",
                )
            },
        }
    except Exception as exc:
        report["mininet_wifi"] = {
            "import_ok": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }

    lowpan_api = bool(report["mininet_wifi"].get("has_addSensor_6lowpan"))
    lowpan_tool = bool(report["commands"].get("iwpan"))
    lowpan_module = report["kernel_modules"]["mac802154_hwsim"]
    lowpan_kernel = bool(
        lowpan_module["currently_loaded"]
        or (lowpan_module["modinfo"] and lowpan_module["modinfo"]["returncode"] == 0)
        or (lowpan_module["modprobe_dry_run"]
            and lowpan_module["modprobe_dry_run"]["returncode"] == 0)
    )
    wwan_api = bool(report["mininet_wifi"].get("has_addModem_wwan"))
    wwan_module = report["kernel_modules"]["wwan_hwsim"]
    wwan_kernel = bool(
        wwan_module["currently_loaded"]
        or (wwan_module["modinfo"] and wwan_module["modinfo"]["returncode"] == 0)
        or (wwan_module["modprobe_dry_run"]
            and wwan_module["modprobe_dry_run"]["returncode"] == 0)
    )

    report["candidate_summary"] = {
        "wifi": {
            "api_present": bool(report["mininet_wifi"].get("has_addStation")),
            "note": "Existing verified access technology.",
        },
        "sixlowpan_ieee802154": {
            "api_present": lowpan_api,
            "iwpan_present": lowpan_tool,
            "kernel_support_detected": lowpan_kernel,
            "candidate_for_live_pilot": lowpan_api and lowpan_tool and lowpan_kernel,
            "note": "A distinct native technology, but live connectivity and mobility still require a separate pilot.",
        },
        "generic_wwan": {
            "api_present": wwan_api,
            "kernel_support_detected": wwan_kernel,
            "candidate_for_interface_pilot": wwan_api and wwan_kernel,
            "note": "Generic WWAN scaffolding is not proof of an LTE/5G radio, RAN, core, or handover.",
        },
    }

    output_path = Path(__file__).resolve().with_name("hosn_cross_tech_report.json")
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("HOSN CROSS-TECH CHECK (READ ONLY)")
    print("Machine: {} | kernel {} | Mininet-WiFi {}".format(
        report["machine"], report["kernel"],
        report["mininet_wifi"].get("version", "unavailable"),
    ))
    print("Wi-Fi API:", "YES" if report["candidate_summary"]["wifi"]["api_present"] else "NO")
    print("6LoWPAN/802.15.4 API:", "YES" if lowpan_api else "NO")
    print("6LoWPAN tool iwpan:", "YES" if lowpan_tool else "NO")
    print("6LoWPAN kernel support:", "YES" if lowpan_kernel else "NO")
    print("Generic WWAN API:", "YES" if wwan_api else "NO")
    print("Generic WWAN kernel support:", "YES" if wwan_kernel else "NO")
    print("LTE/5G claim authorized: NO (this checker does not test a RAN/core)")
    print("Changes made: NO")
    print("Saved detailed report:", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
