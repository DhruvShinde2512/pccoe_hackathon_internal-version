# Case graph and temporal feature registry

Feature version: `case-graph-temporal-v1`. All quantities use visible unique events
in the 48-hour event-time window ending at as_of. Arrival cutoff is enforced first.
IDs, names, labels, split IDs and case IDs are not numerical model inputs.

| Feature | Definition/unit |
|---|---|
| transaction_count | Number of unique transfers |
| total_amount | Sum of transfer amounts; synthetic single-currency units |
| mean_amount | Mean transfer amount; zero for empty context |
| amount_cv | Population amount standard deviation / mean; zero if mean zero |
| account_count | Distinct accounts in transfer endpoints and employee actions |
| unique_senders / unique_recipients | Distinct transfer sender/recipient accounts |
| max_sender_tx_count_1h | Peak same-sender count in inclusive 1-hour event-time windows |
| max_sender_recipients_24h | Peak distinct recipients per sender in inclusive 24-hour windows |
| max_receiver_senders_24h | Peak distinct senders per recipient in inclusive 24-hour windows |
| near_limit_fraction | Fraction of transfers with abs(amount-10000)<=1000; illustrative configuration |
| below_limit_fraction | Fraction of transfers strictly below illustrative limit |
| cycle_closure_count_48h | Closing transfers with a strictly chronological 2/3-edge cycle, one witness each |
| returned_value_fraction_48h | Closing-transfer amount / total transfer amount; not traced provenance of funds |
| max_cycle_length | Longest selected witness, 2 or 3 edges; zero if no cycle |
| employee_count | Distinct employees in visible audit actions |
| access_count | Unique visible audit actions |
| risky_action_count | MODIFY_PHONE/OVERRIDE_ALERT count; actions indicate review context, not guilt |
| transfers_after_risky_action_2h | Transfers preceded within 0–7200 seconds by a risky sender-account action |
| min_risky_action_delta_seconds | Minimum matching action/transfer gap; 172800 sentinel if none |
| risky_access_missing | One when no risky action is visible |
| dormant_sender_fraction | Transfers from supplied dormant accounts / all transfers; missing metadata gives no dormant indication |
| max_accounts_per_employee_24h | Peak distinct accounts touched by one employee in inclusive 24-hour windows |
| case_duration_seconds | Time span between earliest and latest visible evidence |

Every feature returns event IDs supporting its calculation. Ratio denominators
can require all transfer IDs, not only positive examples. Missing account metadata
is returned in API data_quality; dormant inference requires as-of account snapshots.
Self-transfers are not 2/3-edge cycles. Equal timestamps do not establish the
strict ordering necessary for cycle edges. Other count windows include equal times.
No currency conversion is performed: use one currency per benchmark/case or
normalize amounts upstream before applying an illustrative reporting limit.

The rule baseline is intentionally fixed and separately documented in evaluate.py.
It shares these features with the learned models and uses identical test cases.
Case membership discovery, authorization decisions and large graph cycle mining
are outside this feature builder's bounded snapshot contract.
