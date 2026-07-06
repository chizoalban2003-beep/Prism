"""
prism_intents.py
================
Static intent-routing table for PrismAgent: ordered (regex, intent) pairs.
First match wins (see PrismAgent._route). Extracted from prism_agent.py to
keep the agent module focused; behaviour is unchanged.
"""
from __future__ import annotations

INTENTS: list[tuple[str, str]] = [
    # Horizon goals: abandon/list must precede add because the add pattern
    # includes "horizon goal" as a literal trigger, so "list/show my horizon
    # goals" would otherwise register a phantom meta-goal instead of listing.
    # Abandon must precede list because list's "horizon goals?" substring
    # would otherwise eat "cancel that horizon goal".
    (r"(?:stop|cancel|abandon) (?:watching|monitoring|tracking|that horizon|horizon goal)|"
     r"(?:forget|remove|delete) (?:that )?(?:goal|watch|monitor)",
     "horizon_abandon"),
    (r"(?:show|list|what are) (?:my )?(?:horizon|background|watching|monitored) goals?|"
     r"what (?:are you |is prism )?(?:watching|monitoring|tracking)|"
     r"horizon (?:status|goals?|list)",
     "horizon_list"),
    # Horizon add — "when X" / "watch for" / "notify when" must precede every
    # other intent because the trigger clause ("when bitcoin drops") would
    # otherwise be consumed by topic keywords (bitcoin → web_search, etc).
    (r"watch (?:for|out for)|monitor (?:for|when)|track (?:when|until)|"
     r"(?:tell|alert|notify|remind|ping) me when|"
     r"(?:wait|keep watching) (?:for|until)|"
     r"(?:book|buy|do|send|run) (?:it |that )?when |"
     r"horizon goal|long.?term goal|background goal",
     "horizon_add"),
    # Small talk & emotional check-ins: a companion answers "good morning"
    # with a greeting, not a plan card. universal_plan's bare keywords
    # (morning/today) below would otherwise claim these — live probe showed
    # "good morning" and "i feel stressed today" both dead-ending in a
    # "Planner LLM unavailable" error card. Greetings are anchored at BOTH
    # ends (plus optional punctuation/'prism') so "good morning, plan my
    # day" still falls through to the planner; the feelings pattern is
    # unanchored because an emotional statement deserves an empathetic chat
    # reply even mid-sentence. Routing lowercases first — keep patterns
    # lowercase. See issue #28-79.
    (r"^\s*(?:good\s+(?:morning|afternoon|evening|night)|"
     r"hello|hi(?:ya)?|hey(?:\s+there)?|howdy|greetings|yo|"
     r"how(?:'s| is| are) (?:it going|you|things|life|your day)(?:\s+(?:doing|going|today|tonight))*|"
     r"what'?s up|sup|"
     r"thank(?:s| you)(?:\s+(?:so|very)\s+much)?|thankyou|cheers|appreciate (?:it|you)|"
     r"good\s*bye|bye(?:\s+for\s+now)?|see you(?:\s+(?:later|soon|tomorrow))?|"
     r"good\s+job|well done|nice work|love (?:it|that|you))"
     r"[\s!,.?]*(?:prism)?[\s!,.?]*$",
     "general_chat"),
    (r"\b(?:i(?:'m|\s+am)?\s+)?feel(?:ing)?\s+(?:a\s+bit\s+|a\s+little\s+|really\s+|so\s+|very\s+|kinda\s+|quite\s+|pretty\s+)?"
     r"(?:stressed|anxious|sad|down|overwhelmed|tired|exhausted|burn(?:t|ed)[-\s]?out|"
     r"lonely|depressed|worried|nervous|angry|frustrated|upset|low|unmotivated|lost|"
     r"happy|great|good|amazing|excited|energi[sz]ed|motivated|proud)\b",
     "general_chat"),
    # Conversation-history recall — "what did we talk about yesterday",
    # "summarise today's conversation". Time-windowed, answered from the
    # conversation store, not similarity search (which can't match a
    # query that shares no tokens with the stored turns). Must precede
    # universal_plan (bare "today"/"morning" keywords claim these) and
    # memory_recall ("do you remember what we talked about" contains the
    # fact-recall verb cluster). See issue #28-81.
    (r"what (?:did|have) we (?:talk(?:ed)?|chat(?:ted)?|speak|spoke(?:n)?|discuss(?:ed)?)|"
     r"what (?:were|was) we (?:talking|chatting|discussing)|"
     r"what we (?:talked|spoke|chatted) about|"
     r"summari[sz]e (?:our|this|the|today'?s|yesterday'?s) "
     r"(?:conversation|chat|session|discussion)|"
     r"(?:our|the) last conversation|"
     r"what did we do (?:yesterday|today|last week|this week)",
     "conversation_recall"),
    # Live financial/crypto data — must precede plan ("today") and wikipedia ("what is")
    (r"stock (?:price|market|quote)|share price|market cap|"
     r"bitcoin|ethereum|crypto (?:price|market)|coin price|"
     r"(?:price|value) of (?:bitcoin|ethereum|[A-Z]{2,5})\b",
     "web_search"),
    # News must precede plan — "today's headlines" contains "today"
    (r"news|headlines|top stories|latest stories|breaking news", "news_headlines"),
    # Weather must precede plan — "today's weather" contains "today" which
    # universal_plan claims, and weather is more specific. A dedicated
    # entry remains below (line 244) for cases where this hoist doesn't
    # win; ordering still picks the first match.
    (r"\b(?:weather|temperature|forecast|"
     r"how (?:hot|cold|warm)|"
     r"(?:is|will) it (?:rain(?:ing|y)?|sunny|cloudy|hot|cold|warm|chilly|windy"
     r"|snow(?:ing|y)?|hail(?:ing)?|storm(?:ing|y)?|freezing|scorching|foggy)|"
     r"(?:is it|will it) (?:going to |gonna )?"
     r"(?:rain(?:ing)?|snow(?:ing)?|hail(?:ing)?|storm(?:ing)?)|"
     r"(?:will|gonna) (?:it )?(?:rain|snow|hail|storm))\b",
     "weather_check"),
    # Wall-clock queries must precede universal_plan ("today" overlaps).
    (r"^\s*(?:what(?:'s| is)?\s+(?:the\s+)?time(?!\s+(?:is|in|of|for|at|until|till|do)\b)|"
     r"what\s+time\s+(?:is\s+it|do\s+you\s+have)\b|"
     r"current\s+time|time\s+(?:now|please)|"
     r"what(?:'s| is)?\s+(?:today'?s\s+)?date|"
     r"what(?:'s| is)?\s+the\s+date(?:\s+today)?|"
     r"what\s+date\s+is\s+(?:it|today)|"
     r"current\s+date|today'?s\s+date|"
     r"what\s+day\s+is\s+(?:it|today))\b",
     "clock_query"),
    # The (?:in)?to alternation catches "translate X into french" — the
     # previous lookahead only excluded "to french" so "good morning into
     # french" routed here on the "morning" hit. Also bail on any literal
     # "translate" verb in the message, regardless of preposition shape.
     # Reminder/task list queries must precede universal_plan — "today"
     # in "reminders for today" would otherwise be claimed by the planner.
     # Keep this narrow: only LIST verbs + the noun, or interrogatives that
     # specifically ask about pending items. Don't claim "set a reminder".
     (r"\b(?:list|show|view|display|see) (?:my |the |all )?(?:reminders?|tasks?|todos?|to-?dos?)\b"
     r"|^(?:my\s+)?(?:reminders?|tasks?|todos?)\s+(?:for\s+)?(?:today|tomorrow|tonight)\b"
     r"|what (?:reminders?|tasks?|todos?) (?:do i have|are (?:pending|there|due|on|in))",
     "list_tasks"),
     # budget_status must precede universal_plan: phrases like "how much
     # have I spent today" or "spending so far today" contain "today",
     # which would otherwise be claimed by the planner and freeze the
     # chat for 30s on a planner LLM call. Same hoist pattern as
     # list_tasks above.
     #
     # Carefully NOT claiming bare "what's my budget" — that belongs to
     # show_policies (policy engine, distinct concept). Qualifiers
     # (llm/api/prism/daily/monthly) disambiguate to financial spend.
     (r"\bshow\s+(?:me\s+)?(?:my\s+|the\s+)?budget\b|"
     r"\b(?:llm|api|prism)\s+(?:spend|cost|budget|spending)\b|"
     r"\bhow much (?:have (?:i|you)|did (?:i|you)|am i)\s+spen[dt]\w*\b|"
     r"\b(?:spend|spent|spending|costs?)\s+(?:so far\s+)?(?:today|this\s+(?:week|month|day))\b|"
     r"\bremaining\s+(?:budget|spend|credit)\b|"
     r"\b(?:daily|monthly)\s+(?:cost|spend|budget)\b",
     "budget_status"),
     # calendar_read must precede universal_plan — "today" overlaps. Same
     # hoist pattern as budget_status/list_tasks above. Covers the natural
     # phrasings the user types: "any meetings today", "meetings today",
     # "do I have meetings today", "what meetings do I have", and the
     # appointments/events variants.
     (r"\b(?:any|my)\s+(?:meetings?|appointments?|events?)(?:\s+(?:today|tomorrow|this\s+(?:week|month)))?\b|"
     r"\b(?:meetings?|appointments?|events?)\s+(?:today|tomorrow|this\s+(?:week|month))\b|"
     r"\bdo\s+i\s+have\s+(?:any\s+)?(?:meetings?|appointments?|events?)\b|"
     r"\bwhat\s+(?:meetings?|appointments?|events?)\s+(?:do\s+i\s+have|are\s+there)\b|"
     r"\b(?:show|check|view)\s+my\s+(?:agenda|schedule|calendar)\b",
     "calendar_read"),
     # calendar_write must precede universal_plan (steals "schedule") AND
     # browser_task (steals "book"). Hoisted with explicit calendar nouns
     # so "book a table" still routes to browser_task.
     (r"\b(?:schedule|book|create|add|set\s+up)\s+(?:a\s+|an\s+|the\s+)?(?:meeting|appointment|event)\b|"
     r"\b(?:find|when(?:'s| is)\s+(?:the\s+)?next)\s+(?:a\s+|an\s+)?(?:free|available)\s+(?:slot|time)\b",
     "calendar_write"),
     # contacts must precede wikipedia_lookup — line ~416's broad
     # "who is X" / "what is X" pattern otherwise steals "what is John's
     # email" / "who is John Smith". Possessive form unambiguously asks
     # for a person's record, and "(show|find) contact <Name>" /
     # "contact info for <Name>" are the natural verb-noun forms.
     (r"\bwhat(?:'s| is)\s+\w+'s\s+(?:phone(?:\s+number)?|number|email|address|contact)\b|"
     r"\b(?:show|find|get)\s+contact\s+(?:info\s+(?:for\s+)?)?\w+|"
     r"\bcontact\s+info\s+(?:for\s+)?\w+",
     "contacts"),
     # system_lock: local OS screen lock. Hoisted above device_task (which
     # claims "my desktop/computer" file-op verbs) AND smart_home (which has
     # a bare `\b(?:un)?lock\b` for IoT smart locks). Locking the local
     # screen is a hardware-bridge action PRISM should handle directly —
     # via loginctl lock-session — not delegate to a smart home setup the
     # user may not have. See issue #28-68.
     (r"\block\s+(?:my\s+|the\s+)?(?:screen|computer|session|workstation|desktop|laptop|pc|machine)\b",
     "system_lock"),
     # bluetooth_control: local Bluetooth radio. Hoisted above smart_home
     # (whose `turn (?:on|off)` would otherwise treat "turn on bluetooth"
     # as an IoT device). Scoped to the literal noun "bluetooth". See
     # issue #28-73.
     (r"\bbluetooth\s+(?:on|off|status)\b|"
     r"\b(?:turn|switch)\s+(?:on|off)\s+bluetooth\b|"
     r"\b(?:enable|disable)\s+bluetooth\b|"
     r"\bis\s+bluetooth\s+(?:on|off|enabled|disabled)\b",
     "bluetooth_control"),
     # brightness_control: local display brightness. Hoisted above
     # smart_home_control (which claims "dim the lights" — distinct from
     # "dim my screen") and organ_proposal. Scoped to the literal noun
     # "brightness" / "screen brightness", or {brighter,dimmer,dim} with
     # an explicit screen anchor. See issue #28-72.
     (r"\bbrightness\s+(?:up|down)\b|"
     r"\bscreen\s+brightness\s+(?:up|down)\b|"
     r"\b(?:increase|decrease|raise|lower)\s+(?:the\s+|screen\s+)?brightness\b|"
     r"\bset\s+(?:screen\s+)?brightness\s+to\s+\d+|"
     r"\b(?:screen\s+)?brightness\s+to\s+\d+|"
     r"\bmake\s+(?:the\s+|my\s+)?screen\s+(?:brighter|dimmer)\b|"
     r"\bdim\s+(?:my\s+|the\s+)?(?:screen|display|monitor)\b|"
     r"\bbrighten\s+(?:my\s+|the\s+)?(?:screen|display|monitor)\b",
     "brightness_control"),
     # volume_control: local audio volume. Hoisted above spotify_control
     # (matches "volume music" but never bare "volume up") and
     # organ_proposal. Negative-lookahead on the spectrum-tuning verb
     # ("increase verification" → veax) is implicit because we require
     # the literal "volume"/"louder"/"quieter"/"mute" anchor. See issue
     # #28-70.
     (r"\bvolume\s+(?:up|down)\b|"
     r"\bturn\s+(?:up|down)\s+(?:the\s+)?volume\b|"
     r"\b(?:increase|decrease|raise|lower)\s+(?:the\s+)?volume\b|"
     r"\b(?:un)?mute\b|"
     r"\bset\s+volume\s+to\s+\d+|"
     r"\bvolume\s+(?:to\s+)?\d+|"
     r"\bmake\s+it\s+(?:louder|quieter|softer)\b",
     "volume_control"),
     # system_power: suspend / shutdown / restart / logout. Hoisted above
     # browser_task (steals "go to"), policy_inspect (steals "log"), and
     # organ_proposal. Negative-lookaheads exclude app-level scopes so
     # "restart the app" / "shut down this tab" / "sign in" don't get
     # claimed. See issue #28-69.
     (r"\b(?:go\s+to\s+sleep|put\s+(?:my\s+)?(?:computer|laptop|pc|machine|system)\s+to\s+sleep)\b|"
     r"\bhibernate\b|"
     r"\b(?:sleep|suspend)\s+(?:my\s+|the\s+)?(?:computer|laptop|pc|machine|system)\b|"
     r"\b(?:shut\s*down|shutdown)(?!\s+(?:this\s+|the\s+)?(?:tab|browser|app|application|service|window))\b|"
     r"\bpower\s+off\b|"
     r"\bturn\s+off\s+(?:my\s+|the\s+)?(?:computer|laptop|pc|machine|system)\b|"
     r"\b(?:restart|reboot)(?!\s+(?:this\s+|the\s+)?(?:tab|browser|app|application|service|window|process))\b|"
     r"\b(?:log\s*out|logout|sign\s*out|signout|log\s+me\s+out|sign\s+me\s+out)\b",
     "system_power"),
     (r"(?!.*\b(?:in)?to (?:french|spanish|german|japanese|chinese|arabic|russian|hindi"
     r"|italian|portuguese|dutch|korean|turkish|polish|swedish|norwegian|danish|finnish"
     r"|greek|czech|romanian|hungarian|thai|vietnamese|indonesian|hebrew|ukrainian"
     r"|catalan|english)\b)(?!.*\btranslate\b)(?:plan|morning|daily|today|schedule)",
     "universal_plan"),
    (r"how (?:do|can|should) i|plan (?:for|to)|strategy for|"
     r"help me (?:with|plan|reach|achieve|set|accomplish|hit|build|launch|"
     r"finish|complete|start|tackle|prepare)|"
     r"what(?:'s| is) the best way|i want to|i need to|my goal is", "universal_plan"),
    (r"predict|match|fixture|vs|versus", "predict_match"),
    (r"injury risk|squad risk|squad injury|player risk|player fitness|"
     r"\binjury\b|\bsquad\b|\bfitness\b", "squad_risk"),
    # Device inventory: PRISM's core mission is bridging the user to
    # their hardware. Hoisted above all domain matchers because:
    #   * "hardware inventory" was eaten by domain_supply's `inventory`.
    #   * "list my devices" / "show device capabilities" were eaten by
    #     status's bare `device` keyword.
    #   * "what hardware do I have" was eaten by my_profile's
    #     "what (?:do you )?know about me" via the LLM classifier.
    #   * "what is on this computer" was eaten by smart_home's regex.
    # See issue #28-49.
    (r"(?:what|which|list|show|describe)\s+(?:my\s+|the\s+|all\s+)?"
     r"(?:hardware|devices?|capabilities?|"
     r"(?:cli|command[- ]?line)\s+tools?|browsers?)\b|"
     r"\bdevice\s+(?:inventory|capabilities|capability\s+map|scan)\b|"
     r"\bhardware\s+(?:inventory|scan|list)\b|"
     r"what(?:'s| is)?\s+on\s+(?:this|my)\s+(?:computer|machine|laptop|system)\b|"
     r"what\s+can\s+(?:this|my)\s+(?:computer|machine|device)\s+do\b",
     "device_inventory"),
    (r"moment|1v1|keeper|\bshot\b|attack", "moment"),
    (r"session|footage|video|analyse.*play", "session"),
    (r"transfer|market|value|worth", "transfer"),
    (r"triage|chest|pain|fever|symptom|patient", "domain_medical"),
    (r"portfolio|invest|allocation|bonds|equity", "domain_financial"),
    (r"legal|case|litigat|settle|arbitrat", "domain_legal"),
    (r"hire|hiring|recruit|talent|headcount", "domain_hr"),
    (r"supply chain|procurement|inventory|(?:stock|restock) (?:level|order|management)|out of stock",
     "domain_supply"),
    (r"climate|carbon|emission|energy\.policy", "domain_climate"),
    (r"what (?:do you )?know about me|my profile|who am i|crystallise|persona|how well do you know me",
     "my_profile"),
    (r"my (?:week|weekly|month|monthly) (?:report|summary|narrative|review)|"
     r"what happened this (?:week|month)",
     "my_narrative"),
    (r"how (?:much have you |have you )learned|growth report|"
     r"what have you learned about me|prism growth",
     "my_growth"),
    (r"identity|digital\.dna|who\.am", "identity"),
    (r"artifact|past\.decision|what\.have\.i|my artifacts", "artifacts"),
    # Spotify must precede the generic "status" intent — otherwise
    # "spotify status" / "what's playing on spotify" was eaten by the
    # bare-word \bstatus\b regex (returning the system status card) or
    # by wikipedia_lookup's "what is X" catch-all (returning the
    # Wikipedia article on Spotify the company). See issue #28-48.
    (r"(?:play|pause|skip|next|previous|volume|stop)\s+(?:(?:some|a|the|my|that)\s+)?(?:\w+\s+)?(?:music|spotify|song|track|playback)\b|"
     r"what(?:'s| is)?\s+(?:song\s+is\s+)?(?:playing|on)(?:\s+(?:right\s+)?now)?\s+on\s+spotify|"
     r"what(?:'s| is)?\s+playing(?:\s+(?:right\s+)?now)?(?=\s|$|\?)|"
     r"current(?:ly)?\s+playing|now\s+playing|"
     r"\bspotify\s+(?:status|state|now)\b",
     "spotify_control"),
    # hardware_status — pure read-only hardware/system telemetry.
    # Placed *before* generic "status": queries like "battery status",
    # "disk space", "memory usage", "wifi connected", "uptime" are
    # observations, not the daemon-status card. They bypass the
    # approval flow because reading is not actuation. The user's CEO
    # directive frames PRISM as a bridge to hardware components —
    # bridges report state without asking permission first.
    (r"\b(?:battery|disk|memory|ram|cpu|wi-?fi|network|internet|uptime)\b"
     r"(?:\s+(?:level|status|percent|percentage|remaining|life|space|"
     r"free|usage|available|used|left|load|connected|on|working|now))?\b|"
     r"\bfree\s+(?:disk|memory|ram)\s+(?:space|available)?|"
     r"\bhow\s+much\s+(?:disk|memory|ram|cpu)\b|"
     r"\bsystem\s+(?:load|health|stats)\b|"
     r"\bsystem\s+status\b|"
     r"\bhardware\s+(?:status|stats|info)\b|"
     r"\bload\s+average\b|"
     r"\bhow\s+long\s+(?:has\s+)?(?:this|the\s+(?:system|machine))\s+been\s+(?:up|running|on)\b",
     "hardware_status"),
    # Also before generic "status": "git status" is a shell command, not a
    # daemon-status query — the Developer chip sends exactly that text and
    # used to get the "Connected. KDE… KSA…" card back. shell_run keeps the
    # same approval flow as every other shell request.
    (r"\bgit\s+(?:status|log|diff|branch|stash|remote|show)\b", "shell_run"),
    (r"\bstatus\b|connected|device|\bsync\b", "status"),
    # Perception / fused-context snapshot — "what's my current context",
    # "show my context", "perception status". Placed before memory_recall
    # so the catch-all "what's my X" pattern doesn't redirect a perception
    # query into a memory search.
    (r"what(?:'s| is)\s+my\s+(?:current\s+)?context|"
     r"(?:show|tell me|describe)\s+(?:me\s+)?my\s+(?:current\s+)?context|"
     r"^my context\??$|"
     r"\bcurrent context\b|"
     r"\bcontext\s+(?:right now|now)\b|"
     r"perception\s+(?:status|state|snapshot)",
     "current_context"),
    # Pending-approval queue — must precede memory_recall and list_tasks.
    # memory_recall's negative lookahead only fires when the dedicated noun
    # comes immediately after "my", so "what are my pending approvals" would
    # otherwise match the recall verb cluster first.
    (r"(?:list|show|view|see|what(?:'s| is| are)?|any|my)\s+(?:my\s+|the\s+|current\s+|all\s+)?"
     r"(?:pending\s+)?approvals?\b|"
     r"^\s*approvals?\s*\??\s*$|"
     r"pending\s+(?:approval|action)s?\b",
     "approvals_list"),
    # Personal-fact recall — "what is my favourite colour", "do you remember
    # my partner's name". Placed after the specific my_X intents (profile,
    # narrative, growth, identity, artifacts, status) and before the generic
    # wikipedia_lookup/web_search catch-alls, so retrievable personal facts
    # don't get redirected to an encyclopaedia. The negative lookahead lists
    # tokens that have their own dedicated route.
    (r"(?:what(?:'s| is| are)|who(?:'s| is| are)|when(?:'s| is| was)|"
     r"where(?:'s| is| was)|tell me|do you (?:know|remember)|recall|"
     r"what did i (?:say|tell you) about)\s+(?:about\s+)?my\b"
     r"(?!\s+(?:profile|narrative|growth|week|month|tasks?|todos?|to-?do|"
     r"budget|spend|polic|limit|instructions?|rules?|standing|horizon|"
     r"organs?|feedback|inbox|mailbox|email|mail|messages?|calendar|"
     r"schedule|agenda|meetings?|appointments?|events?|files?|downloads?|"
     r"documents?|desktop|pictures?|music|videos?|finances?|transactions?|"
     r"expenses?|health|steps?|sleep|hrv|heart|calories?|artifacts?|"
     r"identity|persona|status|clipboard|contacts?|day|mind|screen|"
     r"context|notes?|approvals?|pending|notifications?|notifs?|alerts?))",
     "memory_recall"),
    (r"index|scan\.files|search\.code|grep|find\.file", "ksa_task"),
    # remove_instruction must accept content between the article and
    # "instruction|rule" — "remove the never mind instruction" and
    # "delete the rule about uber" are natural phrasings the old narrow
    # regex (article *immediately* followed by the keyword) didn't catch.
    # Also hoisted above device_task — without that, "delete the rule"
    # was eaten by device_task's `delete the` file-op branch.
    # See issue #28-50.
    (r"(?:forget|remove|delete)\s+(?:[^.?!\n]+?\s+)?(?:instruction|rule)s?\b|"
     r"\bstop\s+(?:always|never)\b", "remove_instruction"),
    # Word boundaries around the bare file-op verbs are critical: without
    # \b the substring match for "move" inside "remove" sent
    # "remove the never mind instruction" to device_task instead of
    # remove_instruction (so the user couldn't delete a stored
    # instruction from chat — they got an approval card for a no-op).
    # Same for "delete" inside "deletes" etc. See issue #28-50.
    (r"resize|(?:convert|compress) (?:file|image|video)|"
     r"\b(?:rename|move|copy|delete)\b\s+(?:file|folder|directory|the|a|my)|"
     r"\bcreate file\b|"
     r"find file|search (?:in|for)|read file|list files|"
     r"run (?:command|script)|\bexecute\b|open (?:app|file)|"
     r"install (?:package|app)|git (?:commit|push|pull|status)|"
     r"what(?:'s| is) (?:on|in) my(?! screen| calendar| schedule| agenda| inbox| email| mailbox| clipboard)|"
     r"show me (?:my )?files|"
     r"\bmy files?\b|"
     r"\bmy (?:downloads|documents|desktop|pictures|music|videos)\b",
     "device_task"),
    (r"show (?:my )?polic|what(?:'s| are) my (?:budget|polic|limit)|"
     r"current (?:polic|budget|limit)", "show_policies"),
    (r"set (?:my )?(\w+) (?:budget|limit)|auto.?approv|never use|"
     r"require approval|reset (?:all )?polic", "update_policy"),
    (r"(?:running|active|pending|recent) tasks?|task (?:status|progress)|"
     r"what(?:'s| is) (?:running|happening)", "task_status"),
    (r"(?:read|check|show|any|my) (?:new )?(?:emails?|inbox|messages?|mail)|"
     r"unread|what(?:'s| came) in",                        "email_read"),
    (r"(?:send|reply|write|draft) (?:an? )?email|"
     r"email (?:to|them|him|her)",                          "email_send"),
    (r"(?:what(?:'s| is) on my|check my|show) (?:calendar|schedule|agenda)|"
     r"(?:any|my) (?:meetings?|appointments?|events?) (?:today|tomorrow|this week)",
                                                            "calendar_read"),
    (r"(?:schedule|book|create|add) (?:a )?(?:meeting|event|appointment)|"
     r"(?:find|when(?:'s| is) the next) (?:free|available) (?:slot|time)",
                                                            "calendar_write"),
    (r"(?:look up|tell me about|what (?:is|was|are)) .+?(?:on |in )?wikipedia|wikipedia",
     "wikipedia_lookup"),
    (r"(?:go to|open|browse|visit|find (?:on|online)|"
     r"look up(?! .+ (?:on|in) wikipedia)|book|reserve|fill (?:in|out)|"
     r"check (?:the )?(?:price|availability)|what(?:'s| is) (?:on|the) website)",
     "browser_task"),
    (r"show (?:my )?(?:instructions?|rules?|standing orders?)|"
     r"what (?:have you )?(?:remember|know) about my preferences",
     "show_instructions"),
    (r"(?:use|connect|integrate|set up|configure|add) (?:with )?(?!my )(?!the )"
     r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[a-z]+\.[a-z]+)|"
     r"(?:can you|how do i) (?:use|access|connect to) ", "discover_service"),
    (r"search (?:the web|online|internet|for)|"
     r"look up|find (?:out|info|information)|"
     r"what(?:'s| is) (?:the )?(?:latest|current|today)|"
     r"research|who is|where is|when (?:did|does|is)",
     "web_search"),
    # show_notifications must precede send_push: a bare "my notifications"
    # or "any alerts?" is a read intent, not a send intent. The reverse
    # order let send_push's "ping me / alert me" fragments win for
    # phrases like "alerts for me", and "my notifications" fell through
    # to general_chat, where the LLM improvised "Current context".
    (r"(?:what(?:'s| are)?\s+)?(?:my|any|all|the|recent|new)\s+"
     r"(?:notifications?|notifs?|alerts?|reminders?\s+pending)\b|"
     r"(?:show|list|read|view|see|get|check|open)\s+"
     r"(?:my\s+|the\s+|all\s+)?(?:notifications?|notifs?|alerts?)\b|"
     r"^\s*(?:notifications?|notifs?|alerts?)\s*\??\s*$",
     "show_notifications"),
    # send_push: accept "a" or "an", and an optional adjective between
    # the article and the noun ("send me a test notification", "push me an
    # urgent alert"). Pre-fix the regex required "notification" to follow
    # "a " literally, so "a test notification" fell through.
    (r"(?:send|push) (?:me )?(?:an? )?(?:\w+\s+)?(?:notification|alert|reminder)|"
     r"notify me|ping me|alert me",
     "send_push"),
    (r"(?:find|search|look up|who is|contact|call|email) (?:my )?(?:contact|person|colleague|client|friend)",
     "contacts"),
    # note_list must precede note_append: otherwise "list my notes" falls
    # to the LLM classifier which picks note_append and saves the query
    # itself as a new note.
    (r"(?:list|show|read|view|open|see|get|what(?:'s| are)?)\s+(?:my\s+|the\s+|all\s+)?notes?\b|"
     r"my\s+notes\b", "note_list"),
    (r"(?:append|add|write|save|take|jot down|create|start|new|begin) (?:a )?note|note(?:pad)?:? ", "note_append"),
    (r"remind me|set (?:a )?reminder|alert me (?:in|at|when)|"
     r"don't let me forget|in (\d+) (?:minute|hour|day)|"
     r"at (\d+(?::\d+)?(?:am|pm)?)|"
     # Alarms and wake-ups are just reminders with a time. Without
     # this branch "set an alarm for 7am" tried to synthesise a fresh
     # organ when reminder_set already does the job.
     r"set (?:an )?alarm|wake me (?:up )?(?:at|in)",
     "reminder_set"),
    # complete_task must precede add_task and list_tasks — the engine has
    # complete() (with Todoist/GitHub/Linear sync) but nothing routed to
    # it: "remove task X" fell into list_tasks and just echoed the list.
    (r"(?:complete|finish|close|remove|delete|drop|did|done with) "
     r"(?:the |my |a )?(?:task|todo)\b|"
     r"\btask done\b|"
     r"\b(?:done|finished) with\b.{0,50}\b(?:task|todo)\b|"
     r"\bmark (?:the |my |a )?(?:task )?.{0,60}(?:as )?(?:done|complete[d]?|finished)\b",
     "complete_task"),
    (r"(?:add|create|make|new) (?:a )?(?:task|todo|reminder|ticket|issue)|"
     r"(?:i need to|i have to|remember to|don't forget)",
     "add_task"),
    (r"(?:my )?(?:tasks?|todos?|reminders?|to-do|to do|"
     r"what(?:'s| is) (?:on my )?list|pending|backlog|open issues?)|"
     r"(?:list|show|view|see|display) (?:my |the )?(?:tasks?|todos?|reminders?|to-?dos?)|"
     r"what (?:tasks?|reminders?|todos?) (?:do i have|are (?:there|pending))",
     "list_tasks"),
    (r"(?:that was|you were|that(?:'s| is)) (?:wrong|right|too|not|off|correct|"
     r"perfect|bad|good)|(?:i (?:disagree|agree|wouldn't|would)|"
     r"too (?:aggressive|cautious|risky|safe|bold|timid)|"
     r"next time (?:consider|weight|prioritise)|"
     r"(?:more|less) (?:important|weight|focus))", "calibrate"),
    (r"(?:how am i|calibration|what have you learned|"
     r"how (?:accurate|well) (?:are you|is prism)|"
     r"show (?:my )?feedback history)", "calibration_summary"),
    (r"^(?:yes[,.]?|yeah[,.]?|go ahead|approved?|confirm|do it|proceed)[\s!.]*$",
     "approve_pending"),
    (r"^(?:no[,.]?|cancel|stop|don't|abort|never mind)[\s!.]*$",
     "cancel_pending"),
    (r"(?:what tools|learned tools|acquired tools|"
     r"what can you now do|new capabilities|tool list)",
     "list_tools"),
    (r"how did you do that|what steps did you take|"
     r"explain (?:your )?(?:steps?|process|plan)|"
     r"show (?:me )?(?:the )?steps",
     "explain_composition"),
    (r"show (?:me )?(?:the )?chain|chain (?:steps?|history|log)|"
     r"how did (?:the )?chain work|what (?:steps?|logics?) did you use",
     "chain_history"),
    (r"outcome stats?|chain outcomes?|learning stats?|"
     r"completion rate|how (?:often|many chains?) (?:do you )?(?:complete|finish)",
     "outcome_stats"),
    # NOTE: budget_status is hoisted above universal_plan (see line ~73)
    # so that "how much have I spent today" doesn't get claimed by the
    # planner on the bare "today" hit.
    (r"weekly reflection|reflect (?:on (?:this|the|my) )?(?:week|month|today)|"
     r"how (?:did|was) (?:my|the|this) (?:week|month) go|"
     r"weekly summary",
     "reflection"),
    (r"(?:start|begin|enable|use) (?:voice|microphone|listening|speech)|"
     r"(?:stop|disable) (?:voice|listening|speech)|"
     r"(?:voice|speech|microphone) (?:on|off|status|available)|"
     r"(?:transcribe|listen|record) (?:audio|voice|speech|this)",
     "voice"),
    # `help` matched any string containing "help", so "help me reach a goal"
    # got the capabilities card instead of universal_plan. Anchor to standalone
    # forms only: "help"/"?help", "what can/do you do", explicit "commands",
    # "options", "features".
    (r"^\s*help\??\s*$|\bwhat (?:can|do) you do\b|"
     r"\bgive me (?:a |an )?(?:full |complete )?overview of your capabilities\b|"
     r"\b(?:commands|capabilities|features)\b|"
     r"\boptions\b\?", "help"),
    # Organs — loaded capabilities
    (r"(?:what|which|show|list) (?:my )?(?:organs?|loaded (?:capabilities|modules|tools))|"
     r"organ (?:list|status|registry)",
     "list_organs"),
    (r"turn (?:on|off)|set (?:the )?(?:lights?|thermostat|temp)|"
     r"\b(?:un)?lock\b|what(?:'s| is) (?:on|off)(?! (?:my|the) (?:screen|clipboard))|smart home|home assistant",
     "smart_home"),
    # NOTE: split broad email catch-all. Send-shaped phrases must NOT
    # land on email_read — that opens the inbox instead of composing.
    (r"send.*(?:email|mail)|draft.*(?:email|reply)|reply.*email",
     "email_send"),
    (r"(?:check|read|show|open|fetch|get|list).*(?:email|inbox|mail)|"
     r"(?:email|mail).*(?:unread|new|recent)|email.*summary",
     "email_read"),
    # Organ-mapped intents (broad fallback patterns — do not duplicate entries above)
    # Weather routing fully handled by the hoist at line 43; the previous
    # bare `rain|sunny` here caught "haiku about rain" and "sunny day".
    # Specific device/perception organ intents MUST precede the broad
    # wikipedia_lookup catch-all below. The wikipedia pattern matches
    # "what (?:is|was) (?:a |an |the )?[A-Za-z]" which would otherwise
    # hijack "what is on my screen" → encyclopaedia article on Spell
    # checker. Hoisting them keeps the catch-all intact for actual
    # encyclopaedic queries while letting the perception intents win.
    (r"(?:take|capture|grab) (?:a )?screenshot|screenshot", "screenshot_capture"),
    (r"what(?:'s| is) on (?:my |the )?screen|analyse (?:my |the )?screen|"
     r"analyze (?:my |the )?screen|describe (?:my |the )?screen|"
     r"look at (?:my |the )?screen|what do you see|vision query|"
     r"read (?:my |the )?screen|what(?:'s| is) (?:happening|visible) on screen",
     "vision_query"),
    (r"(?:read|what(?:'s| is) on|show|paste|get) (?:my )?clipboard", "clipboard_read"),
    (r"(?:read|open|show|cat|display) (?:my |the )?file|file (?:contents?|read)", "file_read"),
    (r"(?:write|save|create|overwrite) (?:to )?(?:the )?file|write (?:this|that) to", "file_write"),
    # Arithmetic must precede wikipedia_lookup — otherwise "what is the square
    # root of 144" was hitting the broad "what is X" wiki pattern and
    # returning the article on "New Jerusalem". Symbolic operators, the
    # word-form ("plus", "times", ...), and "square root of N" all route here.
    (r"(?:^|\b)(?:calc(?:ulate)?|compute|evaluate|solve)\b\s*\d|"
     r"\d+(?:\.\d+)?\s*(?:\*\*|//|[+\-*/×÷%^])\s*\d|"
     r"\d+\s+(?:plus|minus|times|over|divided\s+by|multiplied\s+by|modulo|to\s+the\s+power\s+of)\s+\d|"
     r"square\s+root\s+of\s+\d|\bsqrt\s*(?:of\s+)?\d|"
     r"\d+(?:\.\d+)?\s+(?:squared|cubed)\b|"
     r"\d+(?:\.\d+)?\s*(?:%|percent)\s+of\s+\d|"
     r"\blog\s+(?:base\s+\d+(?:\.\d+)?\s+)?of\s+\d|"
     r"\bln\s+of\s+\d|"
     r"\bfactorial\s+of\s+\d|"
     r"(?<!\w)\d+!",
     "calc_eval"),
    (r"wikipedia|look up|tell me about|who (?:is|was)|what (?:is|was) (?:a |an |the )?[A-Za-z]",
     "wikipedia_lookup"),
    (r"translate|translation|in (?:spanish|french|german|italian|portuguese|chinese|japanese|arabic|russian|hindi)",
     "translate_text"),
    # Physical-unit disambiguation FIRST: when an unambiguous metric/imperial
    # unit token is present (kg, miles, celsius, …) treat it as a unit
    # conversion even if a currency word like "pounds" also appears
    # ("10 kg to pounds" is weight, not GBP). Currency words (usd, dollar…)
    # are deliberately excluded here so real FX requests fall through.
    (r"\b(?:km|kilometers?|kilometres?|miles?|kg|kilograms?|grams?|mg|"
     r"ounces?|oz|stones?|celsius|fahrenheit|kelvin|meters?|metres?|cm|mm|"
     r"feet|foot|inch|inches|yards?|liters?|litres?|ml|gallons?|pints?|"
     r"mph|kph|km/h)\b[^.]{0,30}?\b(?:to|in|into)\b",
     "unit_convert"),
    (r"(?:convert|exchange|how much) .* (?:usd|eur|gbp|jpy|cad|aud|chf|cny|currency)|"
     r"(?:usd|eur|gbp|jpy|cad|aud|chf|cny) (?:to|in|into)|"
     r"(?:dollar|euro|pound|yen|yuan|franc|rupee|peso|won|ruble|lira|krona|"
     r"baht|ringgit|dirham|real|shekel|zloty|forint|koruna|krone|dinar|"
     r"bitcoin|satoshi|ethereum) (?:to|in|into)|"
     r"(?:convert|exchange) .* (?:dollar|euro|pound|yen|yuan|franc|rupee)",
     "currency_convert"),
    # Negative lookahead skips data-format / code-language conversions
     # ("convert json to yaml", "convert python to javascript") so they
     # fall through to organ synthesis instead of hitting the unit
     # converter and getting back "Could not parse conversion request".
     (r"(?!.*\b(?:json|yaml|yml|xml|csv|tsv|html|markdown|md|toml|ini|sql"
     r"|python|javascript|typescript|ruby|java|rust|golang|kotlin|swift|c\+\+"
     r"|base64|hex|binary)\b)"
     r"(?:(?:convert|how many|how much) .* (?:to|in|into)|"
     r"(?:km|miles|kg|lbs|celsius|fahrenheit|meters?|feet|inches?|liters?|gallons?) (?:to|in|into))",
     "unit_convert"),
    # NOTE: screenshot_capture, vision_query, clipboard_read, file_read and
    # file_write have been hoisted above wikipedia_lookup so the broad
    # "what is X" catch-all doesn't grab "what is on my screen". Only
    # entries unique to this position remain below.
    (r"(?:set|start|create) (?:a )?timer|timer (?:for|of)|countdown", "timer_set"),
    (r"(?:flip|toss)\s+(?:\d+\s+|a\s+|an\s+|me\s+|some\s+|two\s+|three\s+|four\s+"
     r"|five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)?coins?\b|"
     r"coin\s+(?:flip|toss)|"
     r"(?:roll|throw)\s+(?:\d+\s+|a\s+|me\s+|some\s+|two\s+|three\s+|four\s+|five\s+"
     r"|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)?(?:die|dice|d\d+)|"
     r"\broll\s+\d*d\d+\b|"
     r"(?:pick|choose|give me) (?:a )?random (?:number|integer)|"
     r"random number (?:between|from)|"
     r"random (?:choice|pick) (?:from|between|of)",
     "random_pick"),
    # Spotify hoisted to top of file (line ~95) — the control-verb form is
    # narrow enough not to false-positive earlier, and the query form has
    # to beat status/wikipedia_lookup.
    (r"(?:generate|create|make|qr) (?:a )?qr (?:code)?|qr code for", "qr_generate"),
    (r"(?:run|execute|shell|bash|cmd|terminal|command)(?:\s|:)", "shell_run"),
    # phone_call covers both voice calls and SMS — the organ branches on
    # verb (call/phone/ring vs text/sms/message) and handles either path
    # via Twilio. Widened in #28-71 to claim natural forms like "text bob",
    # "sms bob", "send bob a message", "call bob". route_intent lowercases
    # the message, so name matching must use \w/[a-z] not [A-Z].
    (r"(?:make|place|give|dial) (?:a )?(?:phone )?call|"
     r"(?:call|phone|ring) (?:someone|them|him|her|my |the |mom|dad)|"
     r"phone call to|\bcall \d|"
     # text/sms verbs with a name-shaped target. Exclude common noun-
     # phrase tails (text editor / text file / text format / text box).
     r"\btext\s+(?!editor|file|format|document|box|area|message\b)"
     r"(?:someone|them|him|her|mom|dad|me|[a-z]+)\b|"
     r"\bsms\s+\w+\b|"
     # send a text / sms / whatsapp message
     r"\bsend\s+(?:a\s+|an\s+)?(?:text|sms|whatsapp)(?:\s+message)?"
     r"(?:\s+to\s+\w+)?\b|"
     # send <name> a text/sms/message — covers "send Bob a message"
     r"\bsend\s+(?:mom|dad|someone|them|him|her|"
     r"(?!(?:an?|the|my|me)\b)[a-z]+)\s+(?:a\s+)?(?:text|sms|message|whatsapp)\b|"
     # call <name> — narrow proper-name list (lowercased). Common names
     # only, to avoid claiming "call this/that/out/up/over/back/off".
     r"\bcall\s+(?:bob|alice|mom|dad|john|jane|mary|james|mike|sarah|"
     r"emily|david|chris|tom|tim|kate|anna|paul|peter|sam|alex|"
     r"michael|jennifer|robert|linda|patricia|barbara|elizabeth|"
     r"william|richard|charles|joseph|thomas|daniel|matthew|grandma|"
     r"grandpa|aunt|uncle|sis|bro)\b|"
     r"\bphone\s+(?:bob|alice|mom|dad|john|jane|grandma|grandpa)\b|"
     # message <name> — narrow name list to avoid claiming "the message
     # says" / "show me the message".
     r"\bmessage\s+(?:bob|alice|mom|dad|john|jane|mary|james|mike|"
     r"sarah|emily|david|chris|tom|tim|kate|anna|paul|peter|sam|alex|"
     r"michael|grandma|grandpa|aunt|uncle|someone|them|him|her)\b",
     "phone_call"),
    (r"github (?:issue|pr|pull request|repo)|(?:create|list|open) (?:an? )?issue", "github_issue"),
    (r"(?:send|post) (?:a )?(?:message )?(?:to|on) discord|discord", "discord_send"),
    (r"(?:send|post) (?:a )?(?:message )?(?:to|on) telegram|telegram", "telegram_send"),
    (r"(?:control|turn|set|dim) (?:the )?(?:lights?|thermostat|fan|ac|heater|lock|switch)",
     "smart_home_control"),
    (r"my (?:finances?|budget|spending|transactions?|expenses?)|finance (?:summary|report)",
     "finance_summary"),
    (r"my (?:health|steps?|sleep|hrv|heart rate|calories?)|health (?:summary|report|data)",
     "health_summary"),
    (r"(?:brief|briefing|prep|summary) (?:for|before|about) (?:my )?(?:meeting|call|standup)",
     "meeting_brief"),
    (r"(?:overdue|due today|pending|upcoming) (?:tasks?|reminders?|todos?)|task reminder",
     "task_reminder"),
    (r"policy (?:audit|log|history)|audit (?:log|trail)", "policy_audit"),
    # NOTE: entries below are intentionally absent — the following patterns
    # were duplicates of earlier INTENTS entries and have been removed:
    #   news_headlines (subset of the first news entry above)
]
