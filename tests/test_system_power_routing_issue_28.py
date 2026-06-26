"""system_power routing fix for issue #28 bug 68.

Live probes::

  user: "shut down"          → "PRISM intro" general card (no match)
  user: "restart computer"   → organ_proposal ("Build new organ?")
  user: "reboot"             → organ_proposal
  user: "sign out"           → organ_proposal
  user: "log out"            → policy_inspect (claims "log")
  user: "go to sleep"        → browser_task (steals the verb)
  user: "suspend my laptop"  → Approval required (some other organ)

PRISM has no system_power intent — every fundamental local OS power
action (suspend/shutdown/restart/logout) misroutes. This is the heart of
the hardware-bridge gap.

Fix: dedicated ``system_power`` intent + organ. Hoisted above
browser_task, policy_inspect, and organ_proposal. Approval-gated because
shutdown/reboot/logout are work-disruptive.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestSuspendVariants:

    def test_sleep_computer(self):
        assert _route("sleep computer") == "system_power"

    def test_go_to_sleep(self):
        assert _route("go to sleep") == "system_power"

    def test_put_to_sleep(self):
        assert _route("put my computer to sleep") == "system_power"

    def test_suspend_my_laptop(self):
        assert _route("suspend my laptop") == "system_power"

    def test_hibernate(self):
        assert _route("hibernate") == "system_power"


class TestShutdownVariants:

    def test_shut_down(self):
        assert _route("shut down") == "system_power"

    def test_shutdown_computer(self):
        assert _route("shutdown computer") == "system_power"

    def test_power_off(self):
        assert _route("power off") == "system_power"

    def test_turn_off_computer(self):
        assert _route("turn off computer") == "system_power"


class TestRestartVariants:

    def test_restart(self):
        assert _route("restart") == "system_power"

    def test_restart_computer(self):
        assert _route("restart computer") == "system_power"

    def test_reboot(self):
        assert _route("reboot") == "system_power"

    def test_reboot_my_machine(self):
        assert _route("reboot my machine") == "system_power"


class TestLogoutVariants:

    def test_log_out(self):
        assert _route("log out") == "system_power"

    def test_sign_out(self):
        assert _route("sign out") == "system_power"

    def test_logout(self):
        assert _route("logout") == "system_power"

    def test_log_me_out(self):
        assert _route("log me out") == "system_power"


class TestNoOverclaim:

    def test_shut_down_the_browser_tab_not_system_power(self):
        # If user says "shut down the browser tab" we shouldn't claim
        # system_power — but the bare verb "shut down" is unambiguous.
        assert _route("shut down this tab") != "system_power"

    def test_restart_app_not_system_power(self):
        # "restart the app" should not power-cycle the OS.
        assert _route("restart the app") != "system_power"

    def test_log_into_account_not_system_power(self):
        # "log into my account" is auth, not logout.
        assert _route("log into my account") != "system_power"

    def test_signing_in_not_system_power(self):
        assert _route("sign in to spotify") != "system_power"

    def test_lock_screen_still_system_lock(self):
        # system_lock from #28-68 must still claim its scope.
        assert _route("lock screen") == "system_lock"

    def test_smart_home_still_works(self):
        assert _route("smart home") == "smart_home"
