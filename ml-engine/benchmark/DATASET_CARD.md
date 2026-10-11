# ML development benchmark

Seed: 20261010. Cases: 600.

Synthetic development data; not the team's final held-out benchmark or real banking evidence. 60/20/20 case splits with disjoint accounts/employees and January/February/March event periods. Labels are scenario-author assumptions, recorded separately from event payloads. Legitimate treasury, supplier and profile-update look-alikes deliberately overlap suspicious observations. No model or rule output was used to label examples. Both classes can contain dormant accounts. Some employee logs arrive three hours late. No universal legal reporting threshold is assumed.

Each row is a final investigation snapshot. Remove label, scenario, group_id and split before sending to /score-case. Use late events and repeated IDs separately to test incremental revisions. Supply a fresh independently designed dataset from data-engine for final submission.
