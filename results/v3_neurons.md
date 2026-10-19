# v3 results per neuron

Help is the change in log-prob (nats) of the actual next token when the neuron is zero-ablated at the example (positive: the neuron helps). Reliability is how well two halves of a facet agree on what the network does with the neuron there beyond its generic push. The shared-job score is about 1 when two facets share one job and about 0 when unrelated; one job needs 0.5 or more. Promoted tokens are those the neuron raises most on the facet; specific ones are beyond its generic push. The probability line (exploratory, added after the logit-space results) is the KL divergence and share of next-token probability the neuron moves, the cosine between its logit change and the prediction's own log-probs (positive: it sharpens what is already predicted), and the tokens whose probability it raises and lowers most.

| Neuron | Hand-picked | v1 | v2 | v3 (least-related pair) | Median help, A / B | Shared-job score |
|---|---|---|---|---|---|---|
| 1059 | yes | packing-like | packing-like | no use found | -0.0021 / 0.0003 | -0.04 |
| 1641 | yes | packing-like | packing-like | no use found | 0.0 / 0.0005 | -0.002 |
| 3656 | yes | mixed from the start | mixed from the start | no use found | 0.0063 / -0.0 | -0.138 |
| 3812 | yes | mixed from the start | mixed from the start | no use found | 0.0006 / -0.0 | 0.021 |
| 510 |  | packing-like | packing-like | no use found | 0.0002 / 0.0 | 0.067 |
| 2817 |  | packing-like | packing-like | no use found | 0.0001 / -0.0048 | -0.099 |
| 3068 |  | packing-like | packing-like | too few examples | 0.0 / 0.0 | 0.313 |
| 1528 | yes | packing-like | packing-like | no use found | -0.0291 / -0.0005 | -0.013 |
| 93 |  | packing-like | packing-like | no use found | 0.0023 / 0.0004 | 0.022 |
| 3707 | yes | packing-like | packing-like | no use found | -0.0006 / 0.0001 | -0.043 |
| 782 |  | packing-like | packing-like | no use found | 0.0005 / 0.0002 | -0.027 |
| 2976 |  | packing-like | recursive generalization | no use found | 0.0001 / 0.0003 | 0.424 |
| 138 |  | packing-like | packing-like | too few examples | -0.0 / -0.0033 | -0.023 |
| 2346 | yes | packing-like | packing-like | too few examples | 0.0061 / -0.0008 | 0.035 |
| 2570 |  | packing-like | packing-like | no use found | 0.0 / -0.001 | 0.206 |
| 2464 | yes | packing-like | packing-like | no use found | -0.0076 / -0.0002 | 0.027 |
| 2918 | yes | packing-like | packing-like | no use found | -0.0003 / -0.016 | 0.232 |
| 4012 |  | packing-like | packing-like | no use found | -0.0007 / -0.0013 | 0.136 |
| 467 | yes | packing-like | packing-like | no use found | 0.0 / 0.0078 | -0.13 |
| 3686 |  | packing-like | packing-like | no use found | 0.0004 / 0.0053 | -0.066 |
| 4071 | yes | packing-like | packing-like | no use found | -0.0003 / 0.0001 | 0.026 |
| 4094 |  | packing-like | packing-like | no use found | 0.0003 / -0.005 | -0.03 |
| 2442 | yes | packing-like | packing-like | no use found | -0.0054 / -0.0104 | 0.277 |
| 2706 | yes | packing-like | packing-like | no use found | -0.0007 / 0.0 | -0.094 |
| 3282 | yes | packing-like | packing-like | no use found | 0.0003 / 0.0039 | -0.048 |
| 2382 | yes | packing-like | packing-like | no use found | 0.0 / -0.002 | 0.02 |
| 2757 |  | packing-like | packing-like | no use found | -0.0025 / 0.0 | -0.01 |
| 2443 | yes | mixed from the start | mixed from the start | no use found | 0.0003 / 0.0121 | -0.063 |
| 1391 | yes | mixed from the start | mixed from the start | no use found | -0.0008 / 0.0 | 0.15 |
| 61 |  | packing-like | packing-like | no use found | 0.0006 / 0.0008 | -0.081 |
| 4010 |  | packing-like | recursive generalization | no use found | 0.0 / -0.0002 | 0.096 |
| 682 |  | mixed from the start | mixed from the start | no use found | 0.0016 / 0.0014 | -0.02 |
| 1228 |  | packing-like | mixed from the start | no use found | -0.0004 / -0.0 | 0.274 |
| 1345 | yes | packing-like | packing-like | no use found | -0.0029 / -0.0 | -0.018 |
| 3370 | yes | packing-like | packing-like | no use found | 0.0015 / 0.0057 | 0.168 |
| 2019 | yes | packing-like | packing-like | no use found | 0.0 / -0.0023 | -0.061 |
| 1814 | yes | packing-like | packing-like | no use found | -0.0002 / 0.0072 | 0.065 |
| 3915 | yes | packing-like | packing-like | no use found | -0.0014 / -0.0 | -0.033 |
| 4052 |  | packing-like | packing-like | no use found | 0.0003 / -0.0154 | -0.071 |
| 1710 | yes | packing-like | packing-like | no use found | -0.0035 / -0.0004 | 0.064 |

## Neuron 1059: v1 packing-like, v2 packing-like, v3 no use found (facets 1-2)

Facet-typical activation 1.439; patched into random text at that level, its median help is 0.0, KL 0.00041, sharpen cosine -0.005.

**Facet 0, legal: 'See' citation** (40 examples, 14 from v1): harmless, median help -0.001 (p = 0.9629), later predictions 0.0006 (harmless); reliability 0.913, generic share 0.184

- probability: median KL 0.000754, moved 0.015447, sharpen cosine -0.038; raises ' People', ' id', ' United', ' P', ' Id', ' Dom', 'it', ' B'; lowers ',', ' Rule', ' see', ' 42', '1', ' Fed', ' Commonwealth', ' In'
- promoted: 'EVER', 'ías', 'ización', 'ements', '⋯', ' Zoo', ' Ever', ' blo'
- specific promoted: 'EVER', 'osis', '⋯', 'itive', ' друг', ' andere', 'ostic', 'osing'
- specific demoted: 'ctrine', 'rons', 'Before', 'follow', 'mediately', 'enches', 'thems', 'uesday'
- ` testify was⏎⏎  improper, not every comment on a defendant’s choice not to testify⏎⏎  requires reversal.[[ See]] Martinez, 652` → ' Martinez', help 0.013
- ` So. 379 (1905). We note further that the plaintiffs' brief contains no citations to authority supporting their contentions.[[ See]] Rule 28(a` → ' Rule', help -0.172
- ` disturb the factual findings of the district court and magistrate in pretrial hearings unless we determine those findings to be clearly erroneous.    [[See]] United States v.` → ' United', help 0.031

**Facet 1, legal: 'See' citation (2)** (22 examples, 6 from v1): harmless, median help -0.0021 (p = 0.8987), later predictions 0.0004 (harmless); reliability 0.832, generic share 0.232

- probability: median KL 0.000821, moved 0.014974, sharpen cosine -0.078; raises ' id', ' Def', ' be', ' A', ' Id', ' light', 'con', ' Mor'; lowers ',', ' fru', ' pass', ' also', ' 5', ' Laws', ' 28', ' supra'
- promoted: '\\.', 'ización', '⋯', 'cels', ' Indy', '±', 'yset', 'ères'
- specific promoted: '⋯', 'cels', ' genetic', 'idation', ' Indy', 'ères', ' ±', ' di'
- specific demoted: 'enches', 'Before', 'avel', 'bread', 'ear', 'Sen', 'printed', 'appa'
- ` 809 F.Supp. at 368, the usual discretionary limitations on declaratory actions apply unless Congress instructs differently.  [[See]] Franchise Tax` → ' Fr', help 0.117
- `we do not find the meaning of the section so unambiguous or straightforward as to override the command of § 152(b)."[[ See]] Iowa Utilities,` → ' Iowa', help 0.013
- ` noncitizens need only be supported by⏎a rational basis, even where they differentiate on the basis of⏎gender.[[ See]] Fiallo v` → ' F', help -0.009

**Facet 2, travel: price/time ranges** (40 examples, 5 from v1): harmless, median help 0.0003 (p = 0.6656), later predictions -0.0024 (harmless); reliability 0.958, generic share 0.154

- probability: median KL 0.000468, moved 0.010831, sharpen cosine -0.057; raises ';', '4', ' Tue', ' Apr', '9', '10', 'mid', '3'; lowers ' Mon', '6', '11', ')', '5', 'late', '8', ' Sat'
- promoted: 'CP', 'OVID', 'RSP', 'Cos', 'overs', 'JD', 'RP', 'RSA'
- specific promoted: 'RSA', 'OVID', 'JD', 'SRC', 'PCR', 'RSP', 'PSC', 'kins'
- specific demoted: 'afe', ' You', ' AMERICA', ' Your', '"));', "'));", ' Would', 'iab'
- ` it's Friday afternoon. As a guide, a 200-mile trip (eg London to York) will cost £15[[ to]] £30 if you` → ' £', help 0.003
- `; lunch mains £16-17, dinner set menu £30;  hnoon-2.30pm & 5[[-]]10pm Thu-` → '10', help 0.046
- `6.50, mains £10.50-14.50;  hnoon-2.30pm & 5[[-]]9pm)⏎` → '9', help -0.02

- pair 0-1: no use found; score 0.96, specific cosine 0.836, raw cosine 0.869
- pair 0-2: no use found; score -0.026, specific cosine -0.024, raw cosine 0.148
- pair 1-2: no use found; score -0.04, specific cosine -0.036, raw cosine 0.16

## Neuron 1641: v1 packing-like, v2 packing-like, v3 no use found (facets 1-2)

Facet-typical activation 1.359; patched into random text at that level, its median help is 0.0, KL 0.000343, sharpen cosine 0.039.

**Facet 0, periods in formulas/initials** (40 examples, 7 from v1): harmless, median help -0.0 (p = 0.8995), later predictions -0.0 (harmless); reliability 0.637, generic share 0.429

- probability: median KL 8.5e-05, moved 0.003948, sharpen cosine 0.055; raises '15', '\xa0', '31', '30', '5', '39', '29', '4'; lowers '17', '16', '34', 'A', '0', '42', '3', '35'
- promoted: '³', '%"', '{})', ' canonical', "#'", ' leg', ' goddess', ' percent'
- specific promoted: ' combination', ' rapid', ' combined', ' conjunction', ' comparison', ' overlap', ' ratio', ' cum'
- specific demoted: '…]', ']{}.', ']{};', '\\]]{}', '”.', '81', ']]', '18'
- `et al.*, have also reported similar results for Fe$_{1.04}$Te$_{0.6}$Se$_{0[[.]]4}$ and Ba` → '4', help 0.01
- `xFFF2⏎// -0.000562⏎0xFFEE⏎// -0.001396⏎0xFF[[D]]2⏎// -` → '2', help 0.003
- `Math. Stat. 10 (2013), no. 1, 271–291.⏎⏎O.E[[.]] Barndorff-` → '\xa0', help 0.059

**Facet 1, license: 'distributed under'** (18 examples, 7 from v1): harmless, median help 0.0 (p = 0.2288), later predictions -0.0 (harmless); reliability 0.751, generic share 0.295

- probability: median KL 2e-06, moved 1.2e-05, sharpen cosine 0.031; raises ' permission', ' my', ' this', ' express', ' explicit', ' consent', ' GNU', ' My'; lowers ' the', ' prior', ' these', ' our', ' written', ' your', ' terms', ' License'
- promoted: ' Aut', 'Aut', ' GENERATED', 'ing', 'notations', 'iable', 'pages', 'hed'
- specific promoted: 'only', 'lied', 'onant', 'ONLY', 'Aut', 'Only', 'NI', 'them'
- specific demoted: 'oder', 'uction', 'ema', ' Bomb', 'IRST', 'atten', 'xton', 'illin'
- ` author's name, and copyright information. Material from this site may not be collected, printed, or sold in any form[[ without]] specific permission from the` → ' specific', help 0.048
- `*     (C) Copyright Edward Diener 2014.                                    *⏎#  *     Distributed under[[ the]] Boost Software License,` → ' Boost', help -0.0
- `LICENSE-2.0⏎//⏎//Unless required by applicable law or agreed to in writing, software⏎//distributed[[ under]] the License is distributed` → ' the', help 0.0

**Facet 2, fiction dialogue** (40 examples, 4 from v1): harmless, median help 0.0005 (p = 0.2535), later predictions -0.0 (harmless); reliability 0.942, generic share 0.364

- probability: median KL 0.000114, moved 0.004608, sharpen cosine 0.07; raises '"', ' She', ' down', ' “', ' the', 'He', ' He', ' I'; lowers '\n', 'Ver', 'M', ' over', ' all', 'The', 'Mal', ' so'
- promoted: 'aler', 'intervention', ' especial', ' protested', ' Wonder', ' Interest', 'enderer', ' Nice'
- specific promoted: 'aler', ' Product', ' Wonder', ' PARTICULAR', 'intervention', 'resis', 'asper', ' curiosity'
- specific demoted: 'ition', 'cknow', ' apart', 'ifies', ' ourselves', '2', ' yourselves', '?**'
- ` you doing here?" she asked the princess.⏎⏎"I slipped away to rescue Mr. Felicori."⏎[[⏎]]"From who?"` → '"', help 0.033
- ` fangs were long and sharp, and she had an urge to use them. Unexpectedly she felt something soft brush[[ against]] her leg. She` → ' her', help -0.001
- `ex, weakly.⏎⏎"Yes?"⏎⏎"I didn't really get any gold stars at college."⏎[[⏎]]"I know,` → '"', help 0.002

- pair 0-1: no use found; score -0.169, specific cosine -0.117, raw cosine 0.282
- pair 0-2: no use found; score -0.011, specific cosine -0.009, raw cosine 0.39
- pair 1-2: no use found; score -0.002, specific cosine -0.001, raw cosine 0.327

## Neuron 3656: v1 mixed from the start, v2 mixed from the start, v3 no use found (facets 0-2)

Facet-typical activation 1.695; patched into random text at that level, its median help is 0.0, KL 0.00051, sharpen cosine 0.017.

**Facet 0, fiction: 'you'll find'** (40 examples, 8 from v1): harmless, median help 0.0063 (p = 0.2317), later predictions 0.0044 (harmless); reliability 0.927, generic share 0.316

- probability: median KL 0.00095, moved 0.015073, sharpen cosine 0.024; raises ' a', ' himself', ' something', ' another', ' Kal', ' someone', ' myself', ' herself'; lowers ' the', ' him', ' it', ' his', ' them', ' her', ' me', ' he'
- promoted: 'fitting', 'SOL', ' sol', 'sol', 'rig', ' shelter', 'bered', 'orian'
- specific promoted: ' shelter', 'rig', 'fitting', 'protected', 'PAR', ' refuge', 'lod', ' shelters'
- specific demoted: 'amss', ' assertEquals', ' complaints', 'orld', ' violations', ' conflicts', ' controversy', ' suspicions'
- ` all right,” said Denis, smiling. “He’s gone back⏎to Fort Vermilion, and you’ll[[ find]] your camp waiting as` → ' your', help 0.019
- `I borrowed one of your canoes and left it on the⏎shore, half a mile below here. You’ll[[ find]] it waiting.”⏎` → ' it', help 0.002
- `"Albermarle is more intelligent. When I locate him, he's gone, and when I hide, he[[ finds]] me."⏎⏎` → ' me', help -0.053

**Facet 1, fiction: 'found'** (15 examples, 6 from v1): harmless, median help -0.0077 (p = 0.3894), later predictions -0.0026 (harmless); reliability 0.855, generic share 0.27

- probability: median KL 0.001602, moved 0.021032, sharpen cosine 0.053; raises ' myself', ' a', ' ourselves', ' herself', ' another', ' something', ' someone', ' work'; lowers ' her', ' the', ' them', ' that', ' it', ' me', ' Mark', ' couch'
- promoted: 'fitting', 'SOL', 'aced', 'rig', 'secured', ' secure', 'bered', ' sol'
- specific promoted: 'rig', 'protected', 'aced', 'fitting', 'secure', 'cad', 'ré', 'artic'
- specific demoted: ' assertEquals', ' Issue', ' issue', ' antican', ' 08', ' 07', 'IRST', ' Hos'
- ` purring enthusiastically as Kalix produced a can of cat food. Kalix tried to work the can opener, but[[ found]] it too difficult with` → ' it', help -0.019
- ` opened the door. They all filed in and I turned to introduce them to Jackie but she was no longer there. I[[ found]] her by the sink` → ' her', help -0.077
- ` over. No drama. I'd ask Jezmine to find me a job, and if she didn't I'd[[ find]] something on my own` → ' something', help 0.09

**Facet 2, license: 'published by'** (14 examples, 6 from v1): harmless, median help -0.0 (p = 0.8613), later predictions 0.0 (harmless); reliability 0.655, generic share 0.366

- probability: median KL 6e-06, moved 3.3e-05, sharpen cosine 0.035; raises '//', ' even', '*', ' WITHOUT', '  ', ' FOR', ' PROC', '*('; lowers ' BUT', '\n', ' *', ' the', ' more', ' or', 'even', 'the'
- promoted: 'oln', ' /**<', 'ronic', '\\\\\\\\', ' sql', 'iousness', 'ergus', ' Required'
- specific promoted: 'trl', 'ividual', '\\\\\\\\', 'ellig', ' elig', ' privileges', '//!', ' errno'
- specific demoted: 'com', 'is', 'om', 'an', 'o', '\x91', 'at', 'by'
- ` can redistribute it and/or⏎ * modify it under the terms of the GNU Lesser General Public⏎ * License as published[[ by]] the Free Software Foundation` → ' the', help -0.0
- ` * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND⏎ * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING[[,]] BUT NOT LIMITED TO` → ' BUT', help 0.0
- ` redistribute it and/or⏎  * modify it under the terms of the GNU General Public License as⏎  * published[[ by]] the Free Software Foundation` → ' the', help -0.0

- pair 0-1: no use found; score 0.96, specific cosine 0.855, raw cosine 0.896
- pair 0-2: no use found; score -0.138, specific cosine -0.107, raw cosine 0.269
- pair 1-2: no use found; score -0.183, specific cosine -0.137, raw cosine 0.221

## Neuron 3812: v1 mixed from the start, v2 mixed from the start, v3 no use found (facets 0-1)

Facet-typical activation 1.33; patched into random text at that level, its median help is -0.0001, KL 0.00034, sharpen cosine 0.013.

**Facet 0, fiction: 'kill'** (40 examples, 6 from v1): harmless, median help 0.0006 (p = 0.4598), later predictions 0.0012 (harmless); reliability 0.773, generic share 0.602

- probability: median KL 0.000487, moved 0.010414, sharpen cosine -0.007; raises '.', ',', ' smell', ' his', ' there', ' them', ',"', ' off'; lowers ' Sw', ' their', ' and', ' with', ' direction', ' Lisa', ' him', ' in'
- promoted: 'ful', ' tables', ' hers', 'edic', 'ary', ' scenery', 'same', ':*'
- specific promoted: 'ικ', ' roles', ' foli', ' hers', 'ymb', ' grades', 'ologically', 'omi'
- specific demoted: 'ube', '969', ')[@', 'Patients', 'TW', 'ocur', '################', 'users'
- ` matched when it came to gun speed, he thought.⏎⏎What gripped him was revulsion. He was ready to[[ kill]] or be killed over` → ' or', help -0.034
- ` by the hand pulled her up and the fever left her. She began to smile, our eyes were about to burst out[[ of]] our heads in amaz` → ' our', help 0.001
- ` few days before on the back of another man’s motorbike; he decided he was cheating on her, and[[ killed]] her.⏎⏎` → ' her', help -0.004

**Facet 1, legal: paragraph numbers** (22 examples, 5 from v1): harmless, median help -0.0 (p = 0.0059), later predictions 0.0002 (harmless); reliability 0.923, generic share 0.333

- probability: median KL 4e-06, moved 2.9e-05, sharpen cosine 0.082; raises ' and', ' the', ':', ' its', '.', ').', ' -', ' in'; lowers ' omitted', ' that', '     ', ');', '-', ' beyond', ' it', ' an'
- promoted: 'best', 'arms', 'first', 'rest', ' tables', 'ert', ' objects', 'tables'
- specific promoted: 'best', 'objects', ' best', 'hei', ' objects', 'leg', 'top', 'heads'
- specific demoted: ' GS', ' unambig', ')*-', 'iotic', 'EXPORT', 'uran', ' Branch', 'uesday'
- ` statutory requirements of § 254. Second, GTE asserts that the delay in implementation results in an unconstitutional taking.⏎⏎⏎[[189]]⏎i. Stat` → '\n', help -0.0
- ` it to bear the primary responsibility for ensuring the sufficiency of universal service for both interstate and intrastate services.⏎⏎⏎[[274]]⏎These two provisions` → '\n', help -0.0
- ` "specific timetable for completion." See 47 U.S.C. § 254(a)(2).⏎⏎⏎[[19]]⏎2. Schools` → '\n', help -0.0

- pair 0-1: no use found; score 0.021, specific cosine 0.018, raw cosine 0.457

## Neuron 510: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.717; patched into random text at that level, its median help is 0.0, KL 0.00052, sharpen cosine -0.01.

**Facet 0, facet 0** (40 examples, 7 from v1): harmless, median help 0.0002 (p = 0.889), later predictions 0.0002 (harmless); reliability 0.736, generic share 0.628

- probability: median KL 0.000988, moved 0.016787, sharpen cosine -0.004; raises ' seat', ' and', ' to', ' that', ' if', ' want', ' with', ' see'; lowers ' you', ' good', ',', ' great', ' questions', ' nice', ' I', ' of'
- promoted: 'rror', 'erent', 'locking', 'hydration', 'ahl', 'ity', 'rial', ' validity'
- specific promoted: 'ahren', 'iv', 'isted', 'bed', ' ounces', 'reland', 'ived', 'epi'
- specific demoted: 'hee', 'ませ', 'thren', 'LRQ', 'на', 'nar', ' WHEN', 'ío'
- ` no longer.⏎⏎“Yiminy!” he panted, when he stopped. “I am deadt![[ V]]unce in der` → 'un', help 0.018
- ` my braid up. The cool gel on my neck was heavenly, I sighed. "This is pretty red, I[[ hope]] you don't bl` → ' you', help -0.199
- `. His shower hadn't woken him up enough. "Oh you're here, good. Megan, do you think[[ you]] can drive?"⏎` → ' can', help -0.022

**Facet 1, facet 1** (22 examples, 5 from v1): harmless, median help 0.0 (p = 0.5661), later predictions 0.0001 (harmless); reliability 0.838, generic share 0.196

- probability: median KL 1e-05, moved 3.8e-05, sharpen cosine 0.008; raises 'linux', 'string', ' on', 'sys', 'stdlib', 'signal', 'un', '\n'; lowers 'std', 'assert', 'fc', 'int', ' or', 'st', 'errno', ' license'
- promoted: 'ietal', ' activ', ' ident', 'arda', ' activities', 'sites', ' replacements', 'ahead'
- specific promoted: ' assertions', ' defaults', ' implementations', ' replacements', ' activ', ' videos', ' remov', 'arda'
- specific demoted: ' reading', 'al', 'received', 'hin', 'amazon', ' Reading', 'usterity', 'rak'
- `=99:⏎ *⏎ * This Source Code Form is subject to the terms of the Mozilla Public⏎ * License,[[ v]]. 2.0` → '.', help 0.0
- ` If you include any Windows specific code (or a derivative thereof) from⏎ *    the apps directory (application code)[[ you]] must include an acknowled` → ' must', help -0.0
- ` under the License is distributed on an "AS IS" BASIS,⏎  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,[[ either]] express or implied.` → ' express', help -0.0

- pair 0-1: no use found; score 0.067, specific cosine 0.053, raw cosine 0.379

## Neuron 2817: v1 packing-like, v2 packing-like, v3 no use found (facets 0-2)

Facet-typical activation 1.48; patched into random text at that level, its median help is 0.0, KL 0.000392, sharpen cosine 0.024.

**Facet 0, facet 0** (40 examples, 7 from v1): harmless, median help 0.0001 (p = 0.0051), later predictions 0.0001 (harmless); reliability 0.882, generic share 0.553

- probability: median KL 5.3e-05, moved 0.002165, sharpen cosine 0.047; raises ')', '  ', '^-', '-', '                     ', ';', '\n', '--'; lowers '^', 'album', ')\\', '   ', '    ', 'ol', ',', '                      '
- promoted: 'itive', 'iero', 'adin', 'ieri', 'patient', 'PLIED', '名', 'iving'
- specific promoted: 'itive', 'band', 'itives', 'thead', 'isson', 'adier', 'INION', 'IFN'
- specific demoted: '!\\', "]'", ']",', ']{', 'rounds', 'ometers', ' *_', ' Down'
- `⏎  Age ≦ 69/\> 69                                                      65/59                103/140 days (0.[[49]])                       ‐               ` → ')', help 0.06
- `                         4 (3--5)            3 (3--5)              4 (3--5)                  0.[[670]]⏎  Median` → '\n', help 0.0
- `39 ± 3.08 (0.62)37.56 ± 2.17 (0.[[80]])\>80\>` → ')\\', help -0.048

**Facet 1, facet 1** (40 examples, 6 from v1): harmless, median help -0.0 (p = 0.9629), later predictions -0.0001 (harmless); reliability 0.733, generic share 0.482

- probability: median KL 0.000509, moved 0.011239, sharpen cosine 0.002; raises ' him', ' unjust', ' make', ' abuse', ' otherwise', ' probation', ' or', ' nothing'; lowers ' the', ' his', '-', ' err', ' clearly', ' a', 'iting', ' he'
- promoted: 'document', 'j', 'rile', 'fully', 'rug', 'ara', 'oco', 'antry'
- specific promoted: 'document', 'amais', 'iol', 'olt', 'ament', 'ments', 'iu', 'o'
- specific demoted: ' ideology', 'itarian', 'ASH', ' policy', ' command', 'AUTH', ' authority', 'cano'
- ` relied on false and perjured testimony in procuring the convictions against him; and⏎⏎(3) the state[[ denied]] him his right to` → ' him', help 0.035
- `                determined by the Supreme Court of the United States; or⏎⏎                (2) resulted in a decision that was based[[ on]] an unreasonable⏎                ` → ' an', help 0.0
- `⏎primarily for commissioners; in absence of showing that determination was induced⏎by fraud or was wholly arbitrary and founded[[ on]] no adequate determining principal` → ' no', help 0.034

**Facet 2, facet 2** (20 examples, 5 from v1): harmless, median help -0.0048 (p = 0.4749), later predictions 0.0051 (harmless); reliability 0.763, generic share 0.486

- probability: median KL 0.001012, moved 0.017716, sharpen cosine -0.029; raises ' it', ' and', ',', 'que', ' women', ' discrimination', ' even', ' or'; lowers '-', 'man', ' what', ' non', ' that', ' authority', ' like', ' inequ'
- promoted: 'quo', 'kie', 'ara', 'NY', 'abine', 'avage', 'avery', 'riv'
- specific promoted: 'kie', 'quo', 'acs', 'avery', 'abine', 'vious', 'wers', 'pe'
- specific demoted: ' dominance', 'ogen', ' values', ' homogeneity', '──', ' superiority', 'ogenetic', ' forward'
- ` to stick to motherhood, but we have been told that our intellectual or aesthetic creations were inappropriate, inconsequential,[[ or]] scandalous, an` → ' scandal', help -0.067
- ` experienced racism and xenophobia. Disabled people who've found difficulty accessing the same services as the able-bodied.[[ Gender]]-variant folks who` → '-', help -0.025
- ` at all levels, ensure women’s equal rights, access and opportunities for participation and leadership in the economy and to eliminate[[ gender]]-based violence and` → '-', help -0.028

- pair 0-1: no use found; score -0.083, specific cosine -0.066, raw cosine 0.484
- pair 0-2: no use found; score -0.099, specific cosine -0.081, raw cosine 0.479
- pair 1-2: no use found; score 0.35, specific cosine 0.262, raw cosine 0.619

## Neuron 3068: v1 packing-like, v2 packing-like, v3 too few examples (facets 0-2)

Facet-typical activation 2.026; patched into random text at that level, its median help is 0.0, KL 0.000612, sharpen cosine 0.06.

**Facet 0, facet 0** (40 examples, 9 from v1): harmless, median help 0.0 (p = 1.0), later predictions 0.0 (harmless); reliability 0.664, generic share 0.626

- probability: median KL 0.000673, moved 0.01086, sharpen cosine 0.069; raises '_', ' server', '\\', ' get', ' main', ')', '32', '::'; lowers 'error', ' *', ' len', '6', '\n', ' **', 'w', 'buffer'
- promoted: 'INCLUDE', ' UN', 'ALT', 'nat', ' Un', ' nesting', 'United', ' glob'
- specific promoted: 'inction', 'acion', 'sein', 'inn', 'stop', 'spring', 'oddsidemargin', 'Accessor'
- specific demoted: '(#', 'ype', 'YPE', '/((-', 'orect', "...'", 'PY', 'WB'
- `dpaa_fd_list,⏎	dpaa_fd_sg⏎};⏎⏎static inline u[[64]] ldpaa_` → ' ld', help 0.054
- `(ntohl(OPCODE_TID(cmd))))⏎⏎struct tcp_options {⏎	__be[[16]] mss;⏎` → ' m', help 0.016
- ` -3762,7 +3763,7 @@⏎     int port_number;⏎     struct sockaddr_[[in]] client_sai` → ' client', help 0.097

**Facet 1, facet 1** (7 examples, 5 from v1): harmless, median help -0.0001 (p = 0.9375), later predictions 0.0038 (harmless); reliability 0.581, generic share 0.361

- probability: median KL 0.001632, moved 0.023205, sharpen cosine 0.032; raises ' HD', ' 9', ' card', ' 480', ' 7', ' Titan', ' 460', ' 780'; lowers ' 10', ' G', ' R', ' 16', ' 580', ' GT', ' X', ' 8'
- promoted: ' elif', ' Kon', 'kon', 'от', 'moz', 'eme', 'ál', ' Luxem'
- specific promoted: 'olen', '()->', 'eme', 'blo', 'oby', 'ál', ' live', 'ez'
- specific demoted: 'icode', 'iform', 'irectory', 'pires', 'ervice', 'owler', 'QUE', ' secre'
- ` all about the graphics. To that end it supports AMD's latest Radeon RX 480 and Nvidia's GeForce G[[TX]] 10 series graphics cards` → ' 10', help -0.056
- ` load is under the 5% while the CPU is doing all the render work.⏎My GPU is a Nvidia G[[TX]] 745 and my` → ' 7', help 0.065
- ` and NVIDIA PORTAL: 2 pictures on imgur⏎Doesn't Work:⏎#include "stda[[fx]].h"  ` → '.', help -0.0

**Facet 2, facet 2** (7 examples, 5 from v1): harmless, median help 0.0 (p = 0.9375), later predictions 0.0087 (harmless); reliability 0.452, generic share 0.272

- probability: median KL 0.00098, moved 0.009563, sharpen cosine 0.116; raises ' component', '-', ' Native', ' Story', '/', ' native', '.', ' -'; lowers 'JS', ' app', ' App', 'Component', ':', 'js', ' Create', ' JS'
- promoted: "']['", ' full', '___', 'full', 'bootstrap', ' centered', "'?", ',{'
- specific promoted: 'bootstrap', 'zent', "']['", 'xl', 'axon', 'vas', '*>(', 'ietz'
- specific demoted: 'cies', 'onda', 'yland', ' Spacewatch', 'opot', 'orandum', 'Thursday', 'osto'
- ` (and the above 6 minute embedded video) we’ll look at how to add the Storybook style guide to a[[ React]] web app boostrapped` → ' web', help -0.049
- ` assignable to type 'FunctionComponent'.⏎    Type 'Element[]' is missing the following properties from type 'Re[[act]]Element ReactElement Component` → 'Element', help 0.09
- ` New Component Story⏎⏎Now, let's try adding our own story to storybook . Seeing that we just created our[[ React]] app, there aren` → ' app', help -0.062

- pair 0-1: too few examples; score 0.199, specific cosine 0.124, raw cosine 0.536
- pair 0-2: too few examples; score 0.313, specific cosine 0.172, raw cosine 0.502
- pair 1-2: too few examples; score 0.382, specific cosine 0.196, raw cosine 0.447

## Neuron 1528: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 2.141; patched into random text at that level, its median help is -0.0, KL 0.000745, sharpen cosine 0.011.

**Facet 0, 'Mendeleev'** (21 examples, 6 from v1): harms, median help -0.0291 (p = 0.0), later predictions 0.0009 (harmless); reliability 0.982, generic share 0.165

- probability: median KL 0.007554, moved 0.03216, sharpen cosine 0.017; raises 'ley', "'s", ',', 'ck', '-', ' and', ' (', ':'; lowers 'le', 'lev', 'res', 'z', 'zi', 're', 'zes', 'j'
- promoted: 'elen', 'ull', 'onen', 'urt', 'unction', 'lick', 'ports', 'haw'
- specific promoted: 'haw', 'ports', 'aurant', 'avor', 'anded', 'Book', 'ull', 'unction'
- specific demoted: 'g', 'gos', 'gia', 'dn', 'minimum', 'cs', 'gian', 'least'
- ` and 3 cm wide at a pressure of about 0.01 mm of mercury. In 1869, the year of M[[ende]]leev's famous` → 'le', help -0.049
- `century chemistry, though not necessarily as a microscopic explanation. A chemist could be skeptical of atomistic explanations, as M[[ende]]leev and many` → 'le', help -0.013
- ` of, 42–44⏎⏎expansion of, 45⏎⏎isotopy and, 179–182⏎⏎M[[ende]]leev's view` → 'le', help 0.019

**Facet 1, dialogue: 'See you'** (28 examples, 5 from v1): harmless, median help -0.0005 (p = 0.7793), later predictions -0.0038 (harmless); reliability 0.8, generic share 0.516

- probability: median KL 0.000562, moved 0.011679, sharpen cosine 0.023; raises ' and', ' I', ' another', ' of', ' ya', ',', ' this', ' if'; lowers '?"', '."', ' it', ' was', ' that', ' we', '..."', '!"'
- promoted: ' or', '/,', ' eller', 'End', ' или', 'ker', 'nered', ' /**<'
- specific promoted: 'ariju', 'stem', 'pi', 'ims', 'aim', 'thi', 'inea', 'rices'
- specific demoted: 'ILE', 'ean', 'apore', 'e', 'VERTISEMENT', 'alty', 'oler', ' presence'
- `." "Auntie." "Thank you for everything you've done for me until now." "Kohei." "[[See]] you then." "` → ' you', help -0.009
- ` you've nothing else, want to come to my house?" "Is that okay?" "Of course it is." "[[See]] you." "That` → ' you', help -0.0
- ` for who I was." "That's me." "I'm that guy." "I don't think so." "[[See]], I know something` → ',', help 0.016

- pair 0-1: no use found; score -0.013, specific cosine -0.011, raw cosine 0.284

## Neuron 93: v1 packing-like, v2 packing-like, v3 no use found (facets 1-2)

Facet-typical activation 1.534; patched into random text at that level, its median help is -0.0, KL 0.000381, sharpen cosine 0.012.

**Facet 0, facet 0** (40 examples, 8 from v1): harmless, median help 0.0001 (p = 0.8264), later predictions -0.0008 (harmless); reliability 0.98, generic share 0.206

- probability: median KL 7e-05, moved 0.003743, sharpen cosine -0.064; raises '-', 'f', ',', '50', ')', '70', 'ste', '–'; lowers '.', ';', 'tw', '90', '/', '100', '120', '60'
- promoted: 'erdinand', 'ller', 'ös', 'documentclass', 'ogly', '’,', '’', 'nals'
- specific promoted: '\u202f', 'pler', 'ller', 'erdinand', '\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0', '’', 'oster', 'documentclass'
- specific demoted: 'etti', '}"', 'ITEM', 'items', 'ardi', 'elements', '38', '36'
- `.baltasoundhotel.co.uk; Baltasound; s £45-60, d £[[90]]-100;  ` → '-', help 0.004
- `01458-831040; www.glastonburytownhouse.co.uk; Street Rd; d £[[90]]-120)⏎` → '-', help 0.001
- `.lindenguesthouse.co.uk; 22 Linden Ave; d £65-80, f £[[90]]-150;  ` → '-', help 0.005

**Facet 1, facet 1** (40 examples, 7 from v1): harmless, median help 0.0023 (p = 0.0178), later predictions -0.0003 (harmless); reliability 0.66, generic share 0.56

- probability: median KL 0.000303, moved 0.007552, sharpen cosine -0.015; raises ' from', ' of', ' the', ' in', ' at', ' found', ' were', ','; lowers ' on', '.', ' for', ' as', '3', ' would', ' warrant', ' to'
- promoted: 'gets', 'ds', 'ément', 'ze', 'istle', 'ss', 'rums', 'ি'
- specific promoted: 'osphere', ' Premium', ',&', 'IDTH', 'IZE', 'kip', ':&', '\\_'
- specific demoted: ' cytotox', 'ouses', 'onel', 'ovation', 'aler', 'ua', 'sts', 'annot'
- ` United States; or⏎⏎                (2) resulted in a decision that was based on an unreasonable⏎                determination of the[[ facts]] in light of the` → ' in', help -0.0
- ` without jurisdiction,” the resulting judgment is voidable rather than void.⏎⏎Dunbar v. State, 136 Ohio St[[.]]3d 181,` → '3', help -0.014
- `'s arbitrary and capricious standard, Celpage must show that the FCC failed to "articulate[] a rational relationship between the[[ facts]] found and the choice` → ' found', help 0.008

**Facet 2, facet 2** (33 examples, 5 from v1): harmless, median help 0.0004 (p = 0.9578), later predictions -0.0002 (harmless); reliability 0.921, generic share 0.361

- probability: median KL 0.000216, moved 0.007218, sharpen cosine -0.084; raises ')', ' for', ').', '),', ' robe', ' of', ' over', '-'; lowers ' per', ' to', ' wardrobe', ' across', '/', ' or', ';', ' a'
- promoted: 'ller', 'erdinand', 'ocument', '––––––––', '--------------------', ',\\[', 'inally', 'alse'
- specific promoted: ',\\[', '\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0', '\u202f', '\\[[@', 'inho', '\xa0\xa0\xa0\xa0', 'inally', 'ller'
- specific demoted: 'arguments', ' activities', ' events', 'Attr', 'Actions', ' arguments', 'IQ', ' transactions'
- ` 16 awards.⏎⏎Tickets for members are $9 for adults and $5 for children. Nonmembers are $[[11]] for adults and $` → ' for', help 0.01
- `, how does an £18,000-per-night suite on Hyde Park Corner sound? Double rooms ranging between £[[90]] and £180 per` → ' and', help 0.0
- `hies, stand-up paddleboarding and canoeing (each £20). Two-day courses include kayaking (£[[90]]), sailing (£170` → '),', help 0.015

- pair 0-1: no use found; score -0.042, specific cosine -0.034, raw cosine 0.32
- pair 0-2: no use found; score 0.726, specific cosine 0.69, raw cosine 0.764
- pair 1-2: no use found; score 0.022, specific cosine 0.017, raw cosine 0.459

## Neuron 3707: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.407; patched into random text at that level, its median help is 0.0, KL 0.000317, sharpen cosine 0.011.

**Facet 0, travel: 'Top Sights'** (40 examples, 10 from v1): harmless, median help -0.0006 (p = 0.1387), later predictions 0.0028 (harmless); reliability 0.952, generic share 0.16

- probability: median KL 0.000199, moved 0.006676, sharpen cosine 0.055; raises ' &', ' Oxford', ' in', ' a', ' sites', ' resist', ' for', ' main'; lowers '\n', '\n\n', ' the', ' camping', ' it', ' best', ' **', ' spring'
- promoted: ' parap', ' age', '?!', 'scope', 'ât', 'ään', 'ynes', '?).'
- specific promoted: ' age', ' parap', 'zeit', 'ynes', ' aged', 'scope', ' bands', ' depth'
- specific demoted: 'IVER', 'iox', 'assen', 'ucer', ' Manufacturing', 'emas', 'OID', 'oo'
- ` off Brighton's enduring love of the outlandish.⏎⏎Brighton & Hove⏎⏎1Top S[[ights]]⏎⏎1Royal` → '\n', help -0.001
- ` and Cambridge) make Durham an ideal day trip from Newcastle or overnight stop.⏎⏎Durham⏎⏎1Top S[[ights]]⏎⏎1Dur` → '\n', help -0.0
- ` excellent restaurants, and is within easy striking distance of both Edinburgh and Glasgow.⏎⏎Perth⏎⏎1Top S[[ights]]⏎⏎1F` → '\n', help -0.0

**Facet 1, legal: citation commas** (40 examples, 8 from v1): harmless, median help 0.0001 (p = 0.2649), later predictions 0.0001 (harmless); reliability 0.55, generic share 0.502

- probability: median KL 0.000809, moved 0.014724, sharpen cosine 0.032; raises ' p', ' 706', ' and', '.', '1', ' C', ' N', ' 28'; lowers ' pp', ' AS', ' (', ' 702', '\n\n', ' amend', ' District', ' VI'
- promoted: 'llo', ' Said', '~', 'He', 'ellow', 'elig', 'lb', 'ubotu'
- specific promoted: 'I', '~', '---|', 'llo', ' i', 'Dictionary', ' Hidden', 'ellow'
- specific demoted: 'isel', 'anut', 'enin', 'cester', 'lish', 'anne', 'upe', 'ancell'
- ` MILLER & MARY KAY KANE, FEDERAL PRACTICE AND PROCEDURE § 2713[[,]] at 222-23` → ' at', help -0.002
- `arising under the Constitution, laws or treaties of the United States." 28 U.S.C. § 1331[[,]] and in cases in` → ' and', help -0.012
- ` probable cause to instigate the action against Hall, and this was not reversible error. See also Alabama Pattern Jury Instructions[[,]] § 24.05` → ' §', help -0.042

- pair 0-1: no use found; score -0.043, specific cosine -0.031, raw cosine 0.263

## Neuron 782: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.094; patched into random text at that level, its median help is -0.0, KL 0.000244, sharpen cosine 0.007.

**Facet 0, facet 0** (40 examples, 5 from v1): harmless, median help 0.0005 (p = 0.0185), later predictions 0.0 (harmless); reliability 0.719, generic share 0.63

- probability: median KL 0.000123, moved 0.004562, sharpen cosine 0.011; raises ' waiver', ' U', ' its', ' Court', ' Instruction', ' a', '[', ' consecutive'; lowers ',', '\n', ' the', ' Chief', ' Inst', ' to', ' constructions', '.'
- promoted: 'zer', 'ings', 'Gra', 'STS', 'relations', 'inos', 'zes', 'ino'
- specific promoted: 'íc', ' openness', 'erdinand', 'augh', 'Identity', 'grav', 'identity', ' reversible'
- specific demoted: '++,', '++]', 'oche', 'lac', 'camp', 'UK', 'pec', ' /**<'
- ` earlier claim, and because its⏎dismissal for res judicata was not a "final decision" within the purview of[[ 42]] U.S.` → ' U', help 0.029
- ` interests.... Subjecting the day-to-day judgments of prison officials to an inflexible strict scrutiny analysis would seriously[[ ham]]per their ability to` → 'per', help -0.004
- `'s testimony was not essential to the People's case." The use of a witness, such as a probation officer, with[[ all]] of the dangers involved` → ' of', help 0.0

**Facet 1, facet 1** (11 examples, 4 from v1): harmless, median help 0.0002 (p = 0.8311), later predictions 0.0007 (harmless); reliability 0.603, generic share 0.335

- probability: median KL 0.000336, moved 0.011264, sharpen cosine -0.071; raises '(', '            ', ' {', '(){', '#', '        ', ' (', '                '; lowers '()', '                   ', '\n', '                     ', '                    ', '                  ', ' not', ' ()'
- promoted: 'otti', 'itudes', 'ι', 'γ', ' Tut', ' Tig', 'itude', 'ى'
- specific promoted: 'ى', 'itudes', 'scene', 'itude', 'otti', 'uscular', 'idences', 'dip'
- specific demoted: 'never', 'later', ' never', 'afka', ' toujours', 'isle', ' wreck', 'cig'
- `    }⏎⏎    return bitset == 0b11_11111111_11111111_11111111;⏎}⏎⏎void main[[()]] {⏎    assert` → ' {', help 0.056
- ` main function and its parameters have attributes?⏎For example, is this following legal?⏎[[noreturn]] int[[ main]]() { std::` → '()', help -0.041
- ` something like the following:⏎#include <stdlib.h>⏎#include <stdio.h>⏎⏎int[[ main]]( int argc,` → '(', help 0.031

- pair 0-1: no use found; score -0.027, specific cosine -0.018, raw cosine 0.451

## Neuron 2976: v1 packing-like, v2 recursive generalization, v3 no use found (facets 0-1)

Facet-typical activation 1.977; patched into random text at that level, its median help is 0.0, KL 0.000682, sharpen cosine 0.048.

**Facet 0, facet 0** (24 examples, 7 from v1): harmless, median help 0.0001 (p = 0.4223), later predictions 0.0004 (harmless); reliability 0.65, generic share 0.499

- probability: median KL 0.00251, moved 0.016149, sharpen cosine 0.046; raises 'った', 'ア', '�', 'ん', 'フ�', 'じ', '真', '�'; lowers 'ュ', '。', 'ど', 'う', 'り', 'って', 'び', '�'
- promoted: 'ipes', '/*!', 'idium', 'ned', 'ncia', 'Footnote', 'TypeDef', ' hearsay'
- specific promoted: ' apopt', '/*!', 'letcher', 'fle', '^](#', 'SEQU', 'idenote', "{'"
- specific demoted: 'e', 'ilt', 'form', 'gate', 'er', 'itor', 'ру', 'pr'
- `133() {⏎        /*⏎        通过clone方法生成一个对象时，就会不再�[[�]]行构�` → '行', help -0.0
- `器。模式是：⏎⏎+   普通模式：移动光标并�[[�]]行删除` → '行', help -0.001
- `wiki/Vi) 的重新实现。vi 实现了一个非常成功的�[[�]]�念，` → '�', help 0.0

**Facet 1, facet 1** (32 examples, 4 from v1): harmless, median help 0.0003 (p = 0.0498), later predictions 0.0001 (harmless); reliability 0.728, generic share 0.471

- probability: median KL 0.000415, moved 0.002563, sharpen cosine 0.004; raises 'is', ' a', 'ine', 'ichen', 'inia', 'es', ' about', 'ate'; lowers 'ises', 'by', 'ë', 'ulent', ' some', 'ot', 'in', ' the'
- promoted: '){#', 'ipe', 'esta', 'esse', 'ipes', 'ätt', 'essa', 'ään'
- specific promoted: 'FORMAT', 'äs', '****************************************************************', 'éné', 'ethanol', '********************************', 'esta', '){#'
- specific demoted: 'roller', '__,', 'orms', ' nr', 'kins', 'folk', 'ctors', ' ",'
- ` family-friendly way. As well as waxworks populating the private apartments there are jousting tournaments, daily tre[[b]]uchet-f` → 'uc', help -0.0
- ` the Shropshire Hills, the Roaches, the Malvern Hills, Offa's Dyke Path, the T[[iss]]ington Trail and the` → 'ington', help -0.002
- `s and antique shops.⏎⏎The main attraction, however, is the chance to walk or cycle along the **T[[iss]]ington Trail** ,` → 'ington', help 0.005

**Facet 2, facet 2** (19 examples, 5 from v1): harmless, median help 0.0482 (p = 0.0728), later predictions 0.0074 (harmless); reliability 0.876, generic share 0.259

- probability: median KL 0.004826, moved 0.041701, sharpen cosine 0.069; raises 'en', 'im', 'inc', 'men', 'm', 'ki', 'य', 'it'; lowers 'mes', 'il', 'k', '�', 'ur', 'ime', 'kin', 'ili'
- promoted: 'Push', 'ncia', 'esy', 'BA', 'BP', 'Footnote', 'bare', ' booth'
- specific promoted: '^-', 'Broad', 'BA', '80211', 'schemas', ' Corner', 'aird', '^.'
- specific demoted: 'jsp', 'omas', 'ague', 'ctions', 'ame', '\x1d', 'iNdEx', '\x1e'
- `rman Kebabı⏎⏎⏎⏎Pilav⏎⏎⏎⏎Yoğurt⏎⏎⏎⏎Akşam⏎⏎⏎⏎Er[[z]]incan Ç` → 'inc', help 0.234
- `OVA CUMHURİYETİ⏎⏎Toplam Sandık : 2 Toplam Se[[ç]]men : 1.` → 'men', help 0.244
- `N CUMHURİYETİ⏎⏎Toplam Sandık : 6 Toplam Se[[ç]]men : 4.` → 'men', help 0.228

- pair 0-1: no use found; score 0.424, specific cosine 0.292, raw cosine 0.635
- pair 0-2: no use found; score 0.228, specific cosine 0.172, raw cosine 0.464
- pair 1-2: no use found; score 0.192, specific cosine 0.154, raw cosine 0.446

## Neuron 138: v1 packing-like, v2 packing-like, v3 too few examples (facets 1-2)

Facet-typical activation 1.224; patched into random text at that level, its median help is -0.0001, KL 0.000343, sharpen cosine -0.012.

**Facet 0, facet 0** (40 examples, 5 from v1): harmless, median help -0.0001 (p = 0.7146), later predictions -0.0003 (harmless); reliability 0.711, generic share 0.423

- probability: median KL 0.000308, moved 0.008674, sharpen cosine -0.006; raises ' the', 'No', ' AND', ' from', ' an', '          ', ' a', ' its'; lowers ' his', ' pro', 'United', '.', '        ', ',', ' plaintiff', ' and'
- promoted: 'empre', 'vé', 'lé', '--------------------------------------------------------------------------------------------------------------------------------', 'ibo', 'maven', 'ordin', 'érie'
- specific promoted: '---|---|---', 'aterial', 'vé', 'come', 'aring', ' amen', '|_', '------------------------------------------------------------------------'
- specific demoted: 'numbered', 'ile', 'upp', 'onz', 'ioned', 'prototype', 'ordon', 'ounce'
- `tion. We have reviewed the record and the district court's opinions⏎⏎and find no reversible error. Accordingly, we[[ affirm]] on the reason-` → ' on', help -0.001
- ` Circuit Judges.⏎⏎PER CURIAM:*⏎⏎       Sammy K. Shipman, Texas prisoner #583548,[[ appeals]] the district court’` → ' the', help 0.0
- ` STABILE, J.:                            FILED FEBRUARY 26, 2019⏎⏎        Appellant, Prince Isaac,[[ appeals]] from the December 21` → ' from', help 0.023

**Facet 1, facet 1** (9 examples, 5 from v1): harmless, median help -0.0 (p = 0.8203), later predictions 0.0 (harmless); reliability 0.858, generic share 0.053

- probability: median KL 0.0, moved 2e-06, sharpen cosine 0.021; raises ' Neither', '/', ' neither', ' Both', 'Neither', ' All', ' Unless', ' Only'; lowers ' The', '/.', ' Redistributions', ' THE', '\n', '/,', '.', '/_'
- promoted: 'izzard', 'izumab', '![**', 'isor', 'okin', ' 8000', ' 1913', 'untu'
- specific promoted: 'izzard', 'izumab', '![**', 'okin', ' 1913', ' 8000', 'ision', 'isions'
- specific demoted: 'well', 'ット', 'enia', ' criticism', 'mage', 'odot', 'HO', 'vez'
- ` the License.⏎// You may obtain a copy of the License at⏎//⏎// http://www.apache.[[org]]/licenses/LICENSE` → '/', help -0.0
- ` the License.⏎//You may obtain a copy of the License at⏎//⏎//http://www.apache.[[org]]/licenses/LICENSE` → '/', help 0.0
- ` the License. You may obtain a copy of⏎ * the License at⏎ *⏎ * http://www.apache.[[org]]/licenses/LICENSE` → '/', help -0.0

**Facet 2, facet 2** (40 examples, 5 from v1): harmless, median help -0.0033 (p = 0.0972), later predictions 0.0001 (harmless); reliability 0.797, generic share 0.587

- probability: median KL 0.000368, moved 0.009785, sharpen cosine 0.026; raises '"', ' him', ' this', ' it', ' with', ',', ' can', ' the'; lowers '.', 'ed', 'Kal', ' Sw', ' an', 'D', ' a', ' I'
- promoted: 'lei', 'ening', 'rone', 'fefe', ' flavour', 'lé', 'EW', 'hm'
- specific promoted: 'exports', '公', ' formation', ' of', 'SOCK', ' guarantee', ' total', ' we'
- specific demoted: 'tone', 'icit', 'atile', '}{-', 'icer', ' behaviors', 'exe', 'anger'
- `rix!"⏎⏎"Malveria. Is there something wrong?"⏎⏎"Nothing is wrong, Enchantress[[.]] Why do you ask` → ' Why', help -0.045
- ` Decembrius remained human and kept his seat. "I went there to warn him."⏎⏎"W[[arn]] him about what?"` → ' him', help 0.057
- ` water falling on her from the sky. "It will extinguish me. I'll fizzle and die."⏎[[⏎]]"You won't` → '"', help 0.005

- pair 0-1: too few examples; score 0.109, specific cosine 0.085, raw cosine 0.212
- pair 0-2: no use found; score 0.088, specific cosine 0.067, raw cosine 0.531
- pair 1-2: too few examples; score -0.023, specific cosine -0.019, raw cosine 0.164

## Neuron 2346: v1 packing-like, v2 packing-like, v3 too few examples (facets 0-1)

Facet-typical activation 1.285; patched into random text at that level, its median help is 0.0, KL 0.000259, sharpen cosine 0.015.

**Facet 0, French/Portuguese words** (28 examples, 6 from v1): harmless, median help 0.0061 (p = 0.0857), later predictions -0.0013 (harmless); reliability 0.626, generic share 0.511

- probability: median KL 0.000613, moved 0.012003, sharpen cosine 0.02; raises ' dit', ' de', ' como', ' je', ' tu', ' on', ' ?', ' s'; lowers ' plus', 'vez', ' ne', 'ja', ' d', ' demand', 'u', 'h'
- promoted: 'm', 'lad', 'mand', 'l', 'les', 'mys', 'mant', 'la'
- specific promoted: 'm', 'lag', 'rate', 'iels', 'able', 'mable', 'lad', 'emetery'
- specific demoted: 'jen', 'uden', 'ци', 'que', 'QUE', ' errnoErr', 'iten', 'urop'
- ` elle continua et insista jusqu'à ce qu'il eût consenti.⏎⏎— Mais[[ quand]] même, dit l` → ' même', help -0.095
- `.⏎⏎Antes e depois⏎⏎A imagem é tão importante quanto a música...[[ tal]]vez mais. Is` → 'vez', help -0.039
- `ient-ils tous, la belle enfant ! Comme elle est mignonne ! Comme elle est j[[ol]]ie !⏎⏎` → 'ie', help -0.001

**Facet 1, markup: 'color' attribute** (8 examples, 6 from v1): harmless, median help -0.0008 (p = 0.7422), later predictions -0.0008 (harmless); reliability 0.874, generic share 0.177

- probability: median KL 0.000541, moved 0.006699, sharpen cosine 0.017; raises '="#', '=#', '=', '.', ' red', ' black', ' green', ' just'; lowers '="', ' =', '=[', 'Space', '=\\', '={', '="\\', "='"
- promoted: 'Doc', 'avo', 'Document', 'ARG', 'DEFINE', 'prus', 'Liter', ' reader'
- specific promoted: 'prus', 'Doc', '){#', 'ffffffff', 'Document', 'avo', ' reader', 'mathchoice'
- specific demoted: 'queue', 'ull', 'hop', 'op', 'criptor', 'ochrom', ' eff', ' queue'
- `.9"/>⏎            <close/>⏎        </path>⏎        <fillstroke/>⏎        <fillcolor[[ color]]="#4C612` → '="#', help 0.018
- `.2"/>⏎            <close/>⏎        </path>⏎        <fillstroke/>⏎        <fillcolor[[ color]]="#4C612` → '="#', help 0.021
- `35\] node\[above\] [ ]{} (Y3); (Y2) edge \[bend right = 30,[[ color]] = red\] node` → ' =', help -0.002

**Facet 2, decimal points** (34 examples, 5 from v1): harmless, median help 0.0 (p = 0.9808), later predictions 0.0 (harmless); reliability 0.734, generic share 0.44

- probability: median KL 0.000292, moved 0.00665, sharpen cosine 0.052; raises '3', '1', '41', ' Chemical', ' Spect', '57', '56', ' Sort'; lowers '8', '9', ' Struct', ' Comput', ' Let', '6', '46', '44'
- promoted: 'eline', 'gl', 'ken', 'friendly', 'els', 've', 'mat', 'sale'
- specific promoted: ' HOLDERS', 'opf', 'eline', 'от', 'aria', 'jab', 'rized', 'pocket'
- specific demoted: 'OBJ', 'assis', 'orean', '9', '830', 'onucle', '930', ' Fern'
- `)             0.77 (0.39, 1.53)             0.78 (0.39, 1[[.]]56)             0` → '56', help 0.15
- ` pp. 960-968. https://doi.org/10.1016/j.jacc.2012[[.]]04.049⏎` → '04', help -0.068
- `)             0.87 (0.54, 1.39)             0.90 (0.56, 1[[.]]44)             0` → '44', help -0.096

- pair 0-1: too few examples; score 0.035, specific cosine 0.026, raw cosine 0.317
- pair 0-2: no use found; score 0.043, specific cosine 0.029, raw cosine 0.49
- pair 1-2: too few examples; score -0.047, specific cosine -0.038, raw cosine 0.253

## Neuron 2570: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.65; patched into random text at that level, its median help is -0.0, KL 0.000466, sharpen cosine 0.051.

**Facet 0, facet 0** (33 examples, 17 from v1): harmless, median help 0.0 (p = 0.7509), later predictions 0.0002 (harmless); reliability 0.669, generic share 0.425

- probability: median KL 0.00177, moved 0.022392, sharpen cosine 0.035; raises 's', ' voy', ' inform', 'erc', 'ont', 'uce', 'ub', 'pl'; lowers 'f', 'ett', 'p', 'm', 'vant', 'qu', 'ans', ' foi'
- promoted: 'scribe', 'son', 'ERTYPE', ' CURI', 'NESS', 'Sup', 'Cop', 'equation'
- specific promoted: ')}/\\', 'IGEN', 'Sup', 'igner', 'alion', 'binary', ' }_', '�'
- specific demoted: 'ellen', 'ATTR', ' sport', 'wear', '**,**', ' confer', ' Rog', 'STRING'
- ` aimé les accompagner au bal ; aussi alla-t-elle en demander la permission à sa belle[[-]]mère.⏎` → 'm', help -0.058
- ` se hâta de revêtir la robe et alla à la fête des noces. Ni sa belle[[-]]mère, ni` → 'm', help -0.096
- `'apprendre qu'il devait mourir le lendemain matin. Il avait un bon ami en[[ la]] personne du l` → ' person', help 0.215

**Facet 1, facet 1** (40 examples, 7 from v1): harmless, median help -0.001 (p = 0.8264), later predictions -0.0009 (harmless); reliability 0.74, generic share 0.652

- probability: median KL 0.000348, moved 0.0102, sharpen cosine 0.054; raises ' that', ' voice', ' for', 'apt', ' if', ' think', ' breathe', 'unted'; lowers ' I', ' the', ' we', ' where', ' of', 'aped', ' her', ' telling'
- promoted: '>;', 'Patient', ' CURI', 'SIG', ' differently', ' presume', 'FFER', ' recogn'
- specific promoted: '\t', '-->', '»', 'irc', '?_', '</', ';}', '\u2002'
- specific demoted: ' Publishers', 'string', 'ograf', 'mable', 'stringify', 'inals', ' Illustr', 'atum'
- `⏎⏎"Are you sure about that?" Jamie asked with a frown. "You being an unmarried woman and all[[?"]]⏎⏎"They` → '\n', help -0.001
- ` think they're great. Thanks for buying them."⏎⏎Vex was pleased. "I knew you liked them really[[!]] We have to look` → ' We', help -0.047
- `I suppose you're right. You're a lot more experienced at this sort of thing than I am. But I was[[ hoping]] we could make it` → ' we', help -0.039

- pair 0-1: no use found; score 0.206, specific cosine 0.145, raw cosine 0.592

## Neuron 2464: v1 packing-like, v2 packing-like, v3 no use found (facets 0-2)

Facet-typical activation 2.023; patched into random text at that level, its median help is -0.0, KL 0.000725, sharpen cosine 0.028.

**Facet 0, dialogue: closing ?" ,"** (40 examples, 9 from v1): harmless, median help -0.0076 (p = 0.0031), later predictions -0.0017 (harmless); reliability 0.962, generic share 0.35

- probability: median KL 0.000541, moved 0.010824, sharpen cosine 0.077; raises ' he', ' she', ' I', ' She', ' He', ' said', ' Rick', ' Kal'; lowers '\n', ' asked', ' The', ' was', ' "', ' demanded', ' they', ' Mal'
- promoted: ' tying', ' bleeding', ' performing', 'chro', ' developing', 'RELEASE', ' capturing', ' releasing'
- specific promoted: ' bleeding', ' tying', ' developing', ' emerging', ' flowering', ' capturing', ' bloom', ' falling'
- specific demoted: 'well', 'meric', 'erge', ' Certainly', 'ison', 'NESS', 'sover', 'ries'
- ` ever-present terror of her aunt taking her out of college. "You know Aunt Malvie will blame me for this[[?"]]⏎⏎"Sorry` → '\n', help -0.122
- `We sat at the table and he fidgeted.⏎⏎"What's up? Is everything good with your family[[?"]]⏎⏎"Great` → '\n', help -0.029
- `int, and got a⏎jim-dandy load. Goin’ to send any to market to-night[[?”]] Then, casting⏎` → ' Then', help -0.079

**Facet 1, chemistry: alkyl/methyl** (8 examples, 6 from v1): harmless, median help 0.0736 (p = 0.1094), later predictions -0.0015 (harmless); reliability 0.875, generic share 0.315

- probability: median KL 0.003287, moved 0.031, sharpen cosine 0.004; raises 'phosph', 'ene', ' groups', ' and', ' or', ' amine', ',', 'th'; lowers 'bor', '-', ' bor', ' hal', 'al', ' iod', ' group', 'amine'
- promoted: ' ester', 'swick', ' waterm', ' esters', ' Games', ' valign', 'empre', ' play'
- specific promoted: 'empre', 'spi', 'istics', 'LW', 'esti', 'uba', 'swick', ' esters'
- specific demoted: 'augh', 'ower', 'Roll', 'state', 'equivalent', 'Tab', 'yll', 'équ'
- ` In our studies, nevertheless, we have found that the MTB precursors decompose readily at room temperature by reacting with various[[ alkyl]]amines. In [` → 'amines', help 0.055
- ` significantly promote the reaction activity of the phenol *ortho* position, and favor the condensation reaction of *ortho*[[ methyl]]ol or direct formaldehyde` → 'ol', help 0.11
- `Cl6 and Fe2Cl6. Their anhydrous chlorides act as Friedel–Crafts catalysts to introduce[[ alkyl]] groups into aromatic compounds` → ' groups', help 0.093

**Facet 2, chemistry: alkyl/ethyl** (36 examples, 5 from v1): harmless, median help -0.0002 (p = 0.7269), later predictions -0.0006 (harmless); reliability 0.81, generic share 0.526

- probability: median KL 0.000676, moved 0.01037, sharpen cosine 0.027; raises 'hex', 'ened', ' groups', ' in', 'acet', 'ene', 'th', ' ('; lowers '-', ' carb', 'ld', 'transferase', ' of', ')', ' excitation', ' 2'
- promoted: ' ester', ' Games', 'swick', ' esters', ' valign', 'rette', 'ACE', 'garten'
- specific promoted: 'ki', 'ificial', 'flix', 'propylene', 'ilers', 'isher', 'ii', ' lob'
- specific demoted: 'augh', 'opy', 'aines', 'yon', 'ainer', 'Desc', 'Abstract', 'ross'
- `. 156.2--156.8 ppm and 153.4--156.1 ppm were assigned to *para*[[ alkyl]]ated groups and *` → 'ated', help 0.218
- `*)-2-nonenal and the CO~2~ control trap.⏎⏎The release rate of 2-[[ethyl]]hexanol had no` → 'hex', help 0.098
- ` 62.36% C, 5.01% H, 22.52% N.⏎⏎*5-M[[ethyl]]-6-\[` → '-', help -0.034

- pair 0-1: too few examples; score 0.072, specific cosine 0.066, raw cosine 0.376
- pair 0-2: no use found; score 0.027, specific cosine 0.024, raw cosine 0.442
- pair 1-2: too few examples; score 0.768, specific cosine 0.646, raw cosine 0.775

## Neuron 2918: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.93; patched into random text at that level, its median help is 0.0, KL 0.00079, sharpen cosine -0.006.

**Facet 0, legal: § and )(** (40 examples, 18 from v1): harmless, median help -0.0003 (p = 0.5537), later predictions 0.0 (harmless); reliability 0.563, generic share 0.361

- probability: median KL 0.001573, moved 0.022259, sharpen cosine 0.028; raises '1', '2', ' 3', '250', '(', '5', ' general', ' 37'; lowers '3', '4', '\n\n', ' 42', '8', '20', ' 28', ' 8'
- promoted: "—'", '—"', '”—', '—', ')–(', '––', ')[(', '\u202f'
- specific promoted: "—'", '—"', ' tyr', ')–(', ' Payment', 'inical', ' proto', 'validate'
- specific demoted: 'WN', 'GN', 'rf', 'WV', 'ce', 'fort', 'horn', 'agem'
- ` the time fixed in the option within which acceptance could be made."⏎71 Am.Jur.2d, Specific Performance[[ §]] 143 (1973)` → ' 143', help 0.024
- ` of a federal regulation implicated by Count I, Bear Stearns points to SEC Rule 15c3-3(e[[)(]]2), which it` → '2', help 0.024
- ` the first time on appeal.                   See 8 C.F.R. §⏎⏎1003.1(d[[)(]]3)(iv);` → '3', help -0.185

**Facet 1, travel: '£' before prices** (40 examples, 5 from v1): harms, median help -0.016 (p = 0.0208), later predictions -0.0048 (harmless); reliability 0.959, generic share 0.171

- probability: median KL 0.001004, moved 0.018187, sharpen cosine 0.129; raises '5', '25', '45', '18', '15', '19', '60', '20'; lowers '8', '7', '10', '12', '11', '4', '9', '3'
- promoted: '79', ',[', '74', ' **[', '72', '76', '75', '96'
- specific promoted: '79', '85', '72', '96', '82', '84', '87', '75'
- specific demoted: 'ata', 'roup', 'object', 'utes', 'ica', 'quest', 'unda', 'ervice'
- `208-813050; www.bridgebikehire.co.uk; adult £12-15, child[[ £]]6-9;` → '6', help 0.029
- `Budget' is pretty much anything below £100 per night for a double; at the top end, how does an[[ £]]18,000-` → '18', help -0.102
- ` hire centre ( %01433-651261; per half-/full day adult £13/16, child[[ £]]9/11;` → '9', help -0.08

- pair 0-1: no use found; score 0.232, specific cosine 0.17, raw cosine 0.373

## Neuron 4012: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.437; patched into random text at that level, its median help is -0.0, KL 0.000318, sharpen cosine 0.036.

**Facet 0, facet 0** (34 examples, 6 from v1): harmless, median help -0.0007 (p = 0.4365), later predictions 0.0004 (harmless); reliability 0.741, generic share 0.428

- probability: median KL 0.000537, moved 0.00913, sharpen cosine 0.02; raises 'ots', 'ket', 'r', 'êt', 't', 'riers', "'", 'ett'; lowers 're', 'ik', 'oux', 'aste', 'kom', 'ninger', 'res', 'k'
- promoted: 'ker', 'ts', ' themselves', 'kered', 'pendicular', 'fra', 'transform', 'chten'
- specific promoted: 'ts', 'ader', 'te', 'bolt', 'gel', 'lin', 'bottom', 'bol'
- specific demoted: 'inai', '/-/', 'す', 'ACK', ' pace', 'Americ', '.[@', 'Av'
- `mdat ze dat niet deden⏎⏎Czech: ⏎špatně se mnou a přelo[[ž]]ím⏎zá` → 'ím', help -0.01
- `nes sikkerhed. Jeg vil nævne bestemmelserne om forsyningssikker[[hed]], nemlig information` → ',', help -0.009
- `jde mellem medlemsstaterne, med andre ord undgår vi unødvendige[[ overlap]]ninger, vå` → 'ninger', help -0.066

**Facet 1, facet 1** (40 examples, 5 from v1): harmless, median help -0.0013 (p = 0.5578), later predictions 0.0 (harmless); reliability 0.842, generic share 0.41

- probability: median KL 0.000413, moved 0.011631, sharpen cosine 0.051; raises ' stored', ' then', ' analyzed', ' purification', ' concentration', ' identification', ' lig', ' transferred'; lowers ' the', 'rac', ' culture', ' DNA', ' density', ' incubated', '/', ' ATP'
- promoted: 'iking', 'optic', 'enge', 'ippi', 'istas', 'iber', 'kered', 'ennes'
- specific promoted: 'return', 'ood', 'nan', 'range', 'olding', 'rom', '}.', 'OD'
- specific demoted: ' WHEN', ' quando', ' rebuild', 'LY', ')**', 'ruz', 'atform', 'createElement'
- `, cells were lyzed in RIPA buffer (Solarbio, China), and proteins were separated by SDS-PAGE[[ and]] transferred to microporous` → ' transferred', help 0.01
- ` blot analysis {#Sec6}⏎---------------------⏎⏎Total proteins were extracted with RIPA buffer (Thermo Scientific, USA)[[ and]] 1 mM P` → ' 1', help 0.058
- ` CD11b, Ly6C/G (Gr-1), and Ly6G (1A8), fixed[[,]] permeabilized,` → ' perme', help -0.005

- pair 0-1: no use found; score 0.136, specific cosine 0.107, raw cosine 0.481

## Neuron 467: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.449; patched into random text at that level, its median help is 0.0, KL 0.000306, sharpen cosine 0.015.

**Facet 0, subtitles: opening quote** (40 examples, 8 from v1): harmless, median help 0.0 (p = 0.7954), later predictions -0.0004 (harmless); reliability 0.839, generic share 0.492

- probability: median KL 0.000359, moved 0.010475, sharpen cosine -0.048; raises ' time', ' are', ' you', ' I', 'It', 'Oh', ' patient', 'Yeah'; lowers ' go', ' Ed', ' that', 'To', ' an', ' it', ' What', ' From'
- promoted: ' thereof', '-.', '-,', ']*', 'croft', 'soever', ',[@', '-$'
- specific promoted: ' certainly', '-$', 'ws', 'dq', '-"', '-,', 'pw', 'wg'
- specific demoted: ' Reading', ' Problems', ' Need', ' Ow', ' Want', ' Companies', ' Changes', 'zos'
- ` used to this kind of trouble." "Sae." "Yes?" "Here's the information for today's interview."[[ "]]Okay." "Thank` → 'Okay', help 0.025
- ` " Great." " Any luck finding Tio?" " We got a couple of leads." " What do you got?"[[ "]] An address in Colorado` → ' An', help 0.016
- ` yes." " How many carats is this?" " Four!" " Say yes!" " How much did you spend?"[[ "]] This must have cost` → ' This', help 0.004

**Facet 1, 'Supplementary Figure'** (24 examples, 5 from v1): harmless, median help 0.0078 (p = 0.1688), later predictions 0.0007 (harmless); reliability 0.818, generic share 0.385

- probability: median KL 0.000245, moved 0.007058, sharpen cosine 0.016; raises ' the', '(**', ' down', ' 3', ' 4', ' glucose', ' S', ' here'; lowers 'The', ' cisplatin', ' 1', ' MCL', 'SW', ' 2', ' provides', ' that'
- promoted: 'ου', 'obacter', ' now', ' Consortium', ' Freeman', 'cuador', '上', ' themselves'
- specific promoted: 'iben', 'ena', 'enne', ' Wer', ' Wei', ' FW', 'ου', 'enin'
- specific demoted: 'ised', 'orea', 'story', '68', '_[', 'izzard', 'itzer', '20'
- ` viability experiments with Bcl-2/Bcl-XL inhibitors, ABT-263 and ABT-737 ([Supplementary[[ Figures]] 7 and 8](#` → ' 7', help -0.052
- `126C and R337C) cisplatin-resistant ovarian cancer cell line IGROV-1/CDDP ([Supplementary[[ Figure]] S1a and` → ' S', help -0.001
- ` any significant effect on glutathione levels, while combination treatment with APR-246 and cisplatin resulted in more than additive effects ([Supplementary[[ Figure]] S4](#sup` → ' S', help 0.008

- pair 0-1: no use found; score -0.13, specific cosine -0.107, raw cosine 0.375

## Neuron 3686: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.568; patched into random text at that level, its median help is 0.0, KL 0.00044, sharpen cosine 0.029.

**Facet 0, facet 0** (40 examples, 8 from v1): harmless, median help 0.0004 (p = 0.428), later predictions 0.0019 (harmless); reliability 0.695, generic share 0.488

- probability: median KL 0.000426, moved 0.008718, sharpen cosine 0.012; raises 'a', ' the', ' E', 'i', ' benefits', 'La', '1970', ' is'; lowers 'Ct', 'b', ' testify', 'c', '1973', ' right', '1', ' D'
- promoted: ")',", 'insula', "'>", "').", 'ζ', ')</', '!}', 'ttp'
- specific promoted: 'odot', ")',", "').", 'insula', 'φ', 'about', 'policy', 'Policy'
- specific demoted: ' TCP', ' Jun', 'igate', ' Judge', ' Round', 'ésident', ' Ade', 'gue'
- ` of specific negligence. To hold that proof of specific negligence precludes the application of the res ipsa doctrine could lead to the[[ absurd]] result of weak proof` → ' result', help -0.003
- ` substance in New York state court, thus making⏎him inadmissible pursuant to 8 U.S.C. §§ 1182[[(]]a)(2)(` → 'a', help 0.003
- ` of the motion, that the case cannot be decided without examining the entire record. (Brown v. Gow, 126[[ Cal]].App. 113` → '.', help -0.0

**Facet 1, facet 1** (40 examples, 4 from v1): harmless, median help 0.0053 (p = 0.3754), later predictions -0.0005 (harmless); reliability 0.977, generic share 0.102

- probability: median KL 0.001331, moved 0.02142, sharpen cosine -0.006; raises ' Tue', ' Mon', ' Wed', '9', ' have', ' eyes', '10', ' Muse'; lowers ')', ';', ' Sun', ',', ' %', ' Fri', ' daily', ' visit'
- promoted: 'ż', 'DTD', '<!--', 'iert', 'DT', '../', 'Docket', 'icaid'
- specific promoted: 'ibo', 'hos', 'HT', 'oha', 'DLT', 'comput', 'DTD', 'ż'
- specific demoted: 'move', 'ERTYPE', 'chell', 'lez', 'nett', 'CLUDING', 'ROW', ' move'
- ` mind arriving at midnight.⏎⏎The cheapest fares are nonrefundable, so if you miss your train you[['ll]] have to buy a` → ' have', help 0.05
- ` 2-/3-course dinner £21/25;  hnoon-2pm & 6-11.30[[pm]] Tue-Sat)` → ' Tue', help 0.04
- ` afternoon, evening and full-day driving experiences (£350/400/700) take place on select days under the watch[[ful]] eye of a locom` → ' eye', help -0.015

- pair 0-1: no use found; score -0.066, specific cosine -0.054, raw cosine 0.186

## Neuron 4071: v1 packing-like, v2 packing-like, v3 no use found (facets 0-2)

Facet-typical activation 1.945; patched into random text at that level, its median help is 0.0, KL 0.000522, sharpen cosine 0.017.

**Facet 0, travel: hours dash** (40 examples, 15 from v1): harmless, median help -0.0003 (p = 0.6463), later predictions 0.0002 (harmless); reliability 0.973, generic share 0.036

- probability: median KL 0.00164, moved 0.021391, sharpen cosine 0.04; raises '2', '3', '4', '8', '6', 'late', '5', ';'; lowers '11', '10', '1', ' 5', '12', 'Oct', ' 2', ' Oct'
- promoted: 'uity', 'ository', 'ftime', 'body', ' moment', 'assic', 'ци', ' Inventory'
- specific promoted: 'uity', 'ository', 'ftime', ' Inventory', ' moment', 'assic', 'enu', 'body'
- specific demoted: ')**', 'Nor', ')', 'Tal', '")', ' Thr', 'OP', 'T'
- `8am-10.30pm Mon-Thu, to midnight Fri, 10am-midnight Sat, 10am[[-]]4pm Sun;` → '4', help 0.022
- `15;  h8am-8.30pm Mon-Fri, to 9pm Fri & Sat, 9am[[-]]4pm Sun;` → '4', help -0.003
- `  h9.30am-4.30pm Tue-Thu, to 7pm Fri & Sat, 11am[[-]]4pm Sun;` → '4', help 0.006

**Facet 1, license: '(at your option)'** (19 examples, 9 from v1): harmless, median help 0.0 (p = 0.3321), later predictions -0.0 (harmless); reliability 0.784, generic share 0.221

- probability: median KL 9e-06, moved 4.1e-05, sharpen cosine 0.041; raises ' your', ' YOUR', ' provided', ' “', ' give', ' intellectual', ' granted', ' the'; lowers '\n', ' grant', ' any', ' my', ' "', ' their', 'grant', ' for'
- promoted: 'oth', 'orp', ' litres', 'SON', ' Eden', ' Earl', 'othe', 'andra'
- specific promoted: 'oth', ' English', 'othe', 'OLD', ' bgcolor', 'othy', 'unders', 'ours'
- specific demoted: 'cias', 'theless', 'but', 'mask', 'eries', 'pins', 'egin', 'nexpected'
- ` General Public License as published by⏎    the Free Software Foundation, either version 3 of the License, or⏎    ([[at]] your option) any` → ' your', help 0.0
- ` Lesser General Public License as published by the⏎Free Software Foundation; either version 2.1 of the License, or ([[at]] your⏎option)` → ' your', help 0.007
- `  *⏎ *   the Free Software Foundation; either version 2 of the License, or     *⏎ *   ([[at]] your option) any` → ' your', help -0.0

**Facet 2, legal: 'motion for a new trial'** (40 examples, 8 from v1): harmless, median help 0.0001 (p = 0.8057), later predictions 0.0003 (harmless); reliability 0.562, generic share 0.445

- probability: median KL 0.000261, moved 0.006003, sharpen cosine 0.022; raises ' rehearing', 'tw', ' judgment', ' arbitrary', ' new', '.\'"', ' suppress', ' Ma'; lowers ' dismiss', 'mediate', ' hearing', ' Dismiss', ' directed', ' partial', ' writ', ' final'
- promoted: 'olin', 'azed', 'aires', 'équ', ' sentido', 'jo', 'NFTA', 'za'
- specific promoted: 'en', ' conoc', 'idx', 'azi', 'agi', 'vg', 'uggest', 'atch'
- specific demoted: 'rix', 'rica', '~-', 'rium', 'iku', 'ricia', '{},', 'ία'
- ` would not have accepted the juror if a true answer had been given. The Supreme Court there held that the motion for a[[ new]] trial would be denied` → ' trial', help -0.0
- ` per curiam opinion.⏎⏎⏎William Arthur Brown, Appellant Pro Se.    William A. Brafford,⏎Ass[[istant]] United States Attorney,` → ' United', help 0.001
- ` of removal⏎⏎that the Court previously granted in these petitions is⏎⏎VACATED, and any pending motion for[[ a]] stay of removal in` → ' stay', help 0.001

- pair 0-1: no use found; score 0.082, specific cosine 0.071, raw cosine 0.15
- pair 0-2: no use found; score 0.026, specific cosine 0.019, raw cosine 0.14
- pair 1-2: no use found; score 0.103, specific cosine 0.069, raw cosine 0.358

## Neuron 4094: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.538; patched into random text at that level, its median help is 0.0, KL 0.000465, sharpen cosine 0.004.

**Facet 0, facet 0** (40 examples, 10 from v1): harmless, median help 0.0003 (p = 0.39), later predictions 0.0002 (harmless); reliability 0.446, generic share 0.56

- probability: median KL 0.000231, moved 0.004006, sharpen cosine -0.029; raises ' stated', ' where', ' Court', '.', ' was', ' Memorandum', ' from', 'als'; lowers ' to', ' set', ' findings', ' in', 'aled', 'hib', 'ighed', ' Opinion'
- promoted: 'ity', 'NESS', 'aten', 'bbe', ' assertEquals', 'grily', 'iam', 'iab'
- specific promoted: 'idos', 'idas', ' lugar', ' recip', 'iku', 'viv', 'ider', ' strand'
- specific demoted: 'del', 'relim', 'aryng', 'od', ' subsequently', ' disclosures', ' then', 'supplementary'
- `75, 1081⏎(9th Cir. 2018) (overturning prior ruling that rational basis⏎review applied[[ even]] where the relevant statute` → ' where', help 0.078
- ` the Mezzanine⏎Loan’s promissory note waived notice of acceleration, and in turn, shall⏎[[ex]]amine the ultimate tim` → 'amine', help -0.001
- ` to Dismiss, Transfer, or Stay ("Defendants' Supplement"). Upon consideration of the representations of counsel, and for the[[ reasons]] presented below, it` → ' presented', help -0.203

**Facet 1, facet 1** (12 examples, 5 from v1): harmless, median help -0.005 (p = 0.0425), later predictions 0.0004 (harmless); reliability 0.761, generic share 0.395

- probability: median KL 0.000415, moved 0.005686, sharpen cosine 0.001; raises ' function', 'be', ' <', " '<", ',', '(', 'bb', ' {'; lowers ' $.', ' (', ' $', ' $(', 'bt', 'ui', 'br', 'bm'
- promoted: ' discouraged', 'noop', 'ije', 'igue', '�', 'HAND', 'eness', 'NESS'
- specific promoted: 'icates', 'ielle', 'ines', ' protects', ' contributed', 'icus', ' STATES', 'ocamp'
- specific demoted: 'aul', 'EP', 'able', 'EOF', 'oku', 'EM', 'ку', 'course'
- `.props.onDenomChange(e.target.value);⏎    }⏎⏎    render() {⏎        [[return]] (⏎            <` → ' (', help -0.004
- ` 'tylermcginnis' ) userPromise . then ( ( user ) => { const weatherPromise = get[[We]]ather ( user )` → 'ather', help 0.0
- ` "click" , ( ) => { getUser ( "tylermcginnis" ) . then ( get[[We]]ather ) . then` → 'ather', help 0.0

- pair 0-1: no use found; score -0.03, specific cosine -0.017, raw cosine 0.461

## Neuron 2442: v1 packing-like, v2 packing-like, v3 no use found (facets 1-2)

Facet-typical activation 1.956; patched into random text at that level, its median help is -0.0, KL 0.00088, sharpen cosine 0.018.

**Facet 0, dialogue: commands** (40 examples, 8 from v1): harmless, median help -0.0017 (p = 0.1969), later predictions -0.0016 (harmless); reliability 0.845, generic share 0.336

- probability: median KL 0.001234, moved 0.01812, sharpen cosine 0.025; raises ' it', ' the', ' this', ' in', ' here', '.', ' there', ' for'; lowers '."', ' on', ',', ' relax', ' and', ' you', 'é', ' up'
- promoted: ' VERY', ' firmly', 'vre', 'ONT', '/\u200b', '*((', ' tightly', '\\/'
- specific promoted: '��', ' VERY', '*\\', ' firmly', ' EXPRESS', ' ONLY', ' imposs', ' exactly'
- specific demoted: 'home', 'dash', ' commute', ' upgrade', ' learn', 'along', 'join', ' recharge'
- ` gun into me before I got mine out." "I only shot him once, right between the eyes." "Here,[[ feel]]." "Feel` → '."', help -0.021
- `Thank you very much." "There it is." "Reverse it." "Move in on his wrist." "Hold[[ it]] right there." "` → ' right', help -0.027
- `'t even come close," Preacher said. "Your bullet went ten or twelve feet over it, I reckon. Try[[ again]]."⏎⏎By` → '."', help -0.012

**Facet 1, cooking/lab: 'stir'** (23 examples, 7 from v1): harmless, median help -0.0054 (p = 0.7314), later predictions -0.0019 (harmless); reliability 0.861, generic share 0.36

- probability: median KL 0.000988, moved 0.01762, sharpen cosine 0.063; raises ' for', ' on', ' until', ' rod', ' in', ' every', ' into', ' and'; lowers '.', ' with', ' to', ' at', ' by', ' overnight', ' head', ' cream'
- promoted: 'speed', 'bow', 'UES', ' gently', '**_', 'orno', ' speed', '"));'
- specific promoted: 'imod', 'ej', 'speed', 'evin', '♪', ' speed', 'Motion', '|\\'
- specific demoted: 'OPLE', 'your', ' yourselves', 'parable', 'insula', 'parse', 'Your', 'pile'
- ` non-stick saucepan and over fairly low heat, stirring occasionally. Add the semolina. Slowly and rhythmically[[ stir]]-fry the` → '-', help 0.14
- `*g* for 40 min at 5 °C. The membrane-containing pellet was homogenised and then gently[[ stirred]] in solubilisation` → ' in', help -0.025
- ` was increased to 9.0 with sodium bicarbonate, followed by addition of 10 µM benzoyl chloride,[[ incubated]] for 1 h` → ' for', help 0.029

**Facet 2, dialogue: commands (2)** (31 examples, 7 from v1): harmless, median help -0.0104 (p = 0.0664), later predictions -0.0003 (harmless); reliability 0.836, generic share 0.358

- probability: median KL 0.001507, moved 0.019636, sharpen cosine 0.0; raises '."', ' move', ' for', ' on', ' a', ' this', ' in', ' and'; lowers ',', ' up', ' me', ' let', ' it', ' get', '?"', ' shoot'
- promoted: ' VERY', 'oire', 'ONT', 'vre', ' firmly', 'SUCCESS', 'ory', 'posite'
- specific promoted: 'imod', 'TF', 'orate', '---|', ' VERY', '��', 'SR', '#:'
- specific demoted: 'aines', ' expenses', 'cad', 'leman', ' checkout', ' repairs', ' dividends', 'auf'
- `ie." "Are you okay?" "Yeah." "Rosie." "Hey." "Are you okay?" "Come[[ here]]." "What's` → '."', help 0.015
- `Just hold me." "Just please hold me." "All right, babe." " It's okay." " Just[[ hold]] me." " What` → ' me', help -0.01
- ` "This is great." "It is stuck here." "Let's try to pull it out." "Yes." "[[Wait]]~!" "I` → '~', help -0.029

- pair 0-1: no use found; score 0.281, specific cosine 0.24, raw cosine 0.504
- pair 0-2: no use found; score 0.975, specific cosine 0.819, raw cosine 0.882
- pair 1-2: no use found; score 0.277, specific cosine 0.235, raw cosine 0.51

## Neuron 2706: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.297; patched into random text at that level, its median help is -0.0, KL 0.000367, sharpen cosine -0.001.

**Facet 0, fiction: 'surprised'** (40 examples, 12 from v1): harmless, median help -0.0007 (p = 0.6948), later predictions -0.0018 (harmless); reliability 0.91, generic share 0.446

- probability: median KL 0.00025, moved 0.007963, sharpen cosine 0.018; raises ' to', ' when', ',', ' I', ' that', ' she', ' interest', ' speaking'; lowers ' sign', '.', ' by', ' at', ' and', ' anything', ' you', '...'
- promoted: 'avanaugh', 'reams', "'))", '"))', 'etta', ' views', ' sights', 'utta'
- specific promoted: ' sights', 'HECK', ' conclusions', ' review', ' observations', ' judgments', ' recollection', ' insight'
- specific demoted: 'ing', 'protected', 'reated', 'ed', 'ect', 'able', 'omat', 'uffed'
- ` it's obvious anyway. Can't you just stay here and keep on pretending to be normal?"⏎⏎Kalix was[[ surprised]]. Vex's` → '.', help -0.009
- ` didn't intend to make friends with any of her fellow students. She paid little attention to anything the teacher said and was[[ surprised]] when, on hearing` → ' when', help -0.006
- ` you or if you were all right."⏎⏎Savannah caught hold of his extended hands and gaped at him in[[ surprise]]. "Cyrus` → '.', help 0.001

**Facet 1, numbers in math/stats** (40 examples, 5 from v1): harmless, median help 0.0 (p = 0.1831), later predictions -0.0 (harmless); reliability 0.857, generic share 0.375

- probability: median KL 5.5e-05, moved 0.002125, sharpen cosine -0.087; raises '*', '5', '))/(', ' 5', '6', '2', '))/((', ' 0'; lowers '\n', '.', ' +', ' =', ' 6', '-', ' -', '/'
- promoted: 'ETHOD', "'),", '`)', "')", 'caster', "'-", 'casters', 'ilinear'
- specific promoted: 'ilinear', 'ETHOD', 'annotations', '`)', '360', 'tle', '|{\\', 'vphantom'
- specific demoted: '."', 'ank', 'among', ' Doe', '.\\"', '05', 'etic', ' weaker'
- `284, -402011, -578916?⏎-x**3 - 16074*x**2 -[[ 36]]⏎What is the` → '\n', help -0.049
- ` +9512C was found to be significantly associated with the PDR group (OR = 18.45 [2.[[35]], 144.67` → ',', help 0.001
- ` -8103, -19434, -38159?⏎-308*u**3 - u**2 +[[ 72]]*u + 6` → '*', help 0.022

- pair 0-1: no use found; score -0.094, specific cosine -0.083, raw cosine 0.36

## Neuron 3282: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.544; patched into random text at that level, its median help is -0.0, KL 0.000349, sharpen cosine 0.005.

**Facet 0, travel: phone numbers** (40 examples, 5 from v1): harmless, median help 0.0003 (p = 0.2479), later predictions 0.0005 (harmless); reliability 0.917, generic share 0.209

- probability: median KL 0.000688, moved 0.012449, sharpen cosine 0.069; raises ' 8', ' 2', '11', '12', ' 9', ' 4', ' 7', '10'; lowers '-', ' 00', '00', ' 0', ' 66', ' ', ' 67', ' 99'
- promoted: '2', '1', ',', '[,', 'EMPT', 'United', ',[', '3'
- specific promoted: '1', '2', 'X', 'United', 'PUBL', ',', '[,', '4'
- specific demoted: 'ement', 'ament', ' Resolution', 'ilst', 'rophy', '======================', 'emente', 'imension'
- ` train and coach tickets can be very cheap.⏎⏎### INFORMATION SERVICE⏎⏎Traveline ( %0871[[ 200]] 2233; www` → ' 2', help 0.119
- ` islands in the Firth of Clyde.⏎⏎Northlink FerriesFERRY⏎⏎( %0845[[ 600]] 0449; www` → ' 0', help -0.046
- `trentbarton.co.uk) buses provide the majority of bus services. See Traveline ( %0871[[ 200]] 2233; www` → ' 2', help 0.083

**Facet 1, fiction: 'behind her'** (40 examples, 5 from v1): harmless, median help 0.0039 (p = 0.5099), later predictions -0.0006 (harmless); reliability 0.826, generic share 0.465

- probability: median KL 0.000679, moved 0.013153, sharpen cosine -0.026; raises '.', ',', ' and', 's', 'nia', ' that', ' tone', 'algia'; lowers 'ia', ' mood', 'rils', ' bed', ' face', ' well', ' raced', ' she'
- promoted: 'verb', ' unlock', 'ANS', 'VEN', ' again', ' yourself', 'ords', ' itself'
- specific promoted: ' ourselves', 'key', 'verb', ' unlock', ' yourself', 'gov', ' yourselves', 'avis'
- specific demoted: 'toolt', 'Managed', 'itudes', 'FIG', 'USB', ' newest', ' tall', 'ICES'
- ` Kabachetka's spell, but she had refused to show any sign of suffering. She'd heard some noise behind[[ her]], as if other` → ',', help 0.002
- `"Stupid college," she muttered and disappeared upstairs to her room.⏎⏎Vex bounded into the living room behind[[ her]].⏎⏎"` → '.', help 0.009
- ` fur. It began to meow. Kalix sighed with frustration and marched into the kitchen. The cat bounded along behind[[ her]], purring enthusi` → ',', help 0.004

- pair 0-1: no use found; score -0.048, specific cosine -0.042, raw cosine 0.284

## Neuron 2382: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.389; patched into random text at that level, its median help is 0.0, KL 0.000376, sharpen cosine 0.048.

**Facet 0, numbers in technical data** (40 examples, 5 from v1): harmless, median help 0.0 (p = 0.2317), later predictions -0.0002 (harmless); reliability 0.875, generic share 0.67

- probability: median KL 3.8e-05, moved 0.001566, sharpen cosine 0.014; raises 'M', 'A', '^', '6', '+,', ')\\<', '        ', 'C'; lowers 'D', '5', '^--', '.', '\n', '\t', '7', 'I'
- promoted: 'vp', 'vt', 'oL', 'kel', ' NV', 'aine', ' Lemon', ' jul'
- specific promoted: 'kov', 'hov', 'ku', ' another', 'uppose', ' 406', 'bourne', 'kh'
- specific demoted: 'RESULTS', 'BS', 'SD', 'rd', 'IE', '®', 'RD', 'BC'
- `setminus \\{v\\}\)$$⏎](A271843_1_En_12_Chapter_IEq[[84]].gif), otherwise` → '.', help -0.0
- `19	79607	95.3125	+	.	seq2;seq2Range=77955,[[78]]143;eValue` → '143', help -0.004
- `2Range=136934,137018;eValue=5.89745e-28;cigar=[[82]]M1D3` → 'M', help 0.001

**Facet 1, pregnancy/obstetrics** (13 examples, 5 from v1): harmless, median help -0.002 (p = 0.0171), later predictions -0.0029 (harmless); reliability 0.712, generic share 0.547

- probability: median KL 0.000338, moved 0.009447, sharpen cosine 0.087; raises 's', ' units', ' Int', ' education', ' wards', ' unit', ' clinics', ' or'; lowers 'ian', ' and', '\n', 'ians', '-', ' explained', ' {#', ' Outcomes'
- promoted: 'LE', 'FORMAT', 'LET', ' Le', ' Kur', 'reek', 'arine', 'le'
- specific promoted: 'inese', 'leave', ' lazy', 'iners', 'FORMAT', 'inent', 'ropolitan', 'lei'
- specific demoted: ' horiz', 'stem', ' SOFTWARE', 'sky', 'cul', 'ci', 'uy', 'rup'
- `, alleviate their mental pain and reduce stress, depression or anxiety is outside their field of competence as gynaecologists and[[ obstetric]]ians, an area` → 'ians', help -0.002
- ` behaviors in pregnant women.[@CIT0001] In a study conducted by Thaewpia[@CIT0032] on older[[ pregnant]] women, there was` → ' women', help -0.002
- ` to suffer a state of shock, in which she is unable to process the information provided. A nursing assistant who works in[[ pregnancy]] monitoring consultations and who` → ' monitoring', help -0.091

- pair 0-1: no use found; score 0.02, specific cosine 0.015, raw cosine 0.612

## Neuron 2757: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.145; patched into random text at that level, its median help is -0.0, KL 0.000311, sharpen cosine 0.036.

**Facet 0, facet 0** (40 examples, 6 from v1): harmless, median help -0.0025 (p = 0.0153), later predictions -0.0003 (harmless); reliability 0.574, generic share 0.507

- probability: median KL 0.000282, moved 0.008062, sharpen cosine 0.043; raises ' clear', ' reversal', ' or', ' doctrines', ' best', ' will', ' defined', ' counsel'; lowers ' weight', ' liable', ' classification', '\n', ' understood', ' reversing', ' attention', ' appellant'
- promoted: 'ism', 'imi', 'ike', 'MI', 'istic', 'Again', ' interrog', 'aye'
- specific promoted: 'arian', 'arp', 'ite', 'iche', 'haw', 'itt', '…]', 'ikes'
- specific demoted: 'face', 'faced', 'Face', 'FACE', ' face', 'resident', 'B', 'occo'
- `published DispositionNOTICE: Sixth Circuit Rule 24(c) states that citation of unpublished dispositions is disfavored except[[ for]] establishing res judicata,` → ' establishing', help -0.0
- ` that he had probable cause.⏎Defendant City of Lancaster is also not liable for⏎Congress did not intend municipalities to be[[ held]] liable unless action pursuant` → ' liable', help -0.017
- ` it, the court misinterpreted the effect of the evidence, or⏎this court is convinced that the findings are against[[ the]] preponderance of⏎cred` → ' preponderance', help -0.014

**Facet 1, facet 1** (20 examples, 5 from v1): harmless, median help 0.0 (p = 0.498), later predictions 0.0002 (harmless); reliability 0.693, generic share 0.394

- probability: median KL 0.000205, moved 0.004685, sharpen cosine 0.02; raises 'Group', '>(', ' className', '="">', '="', '="\'', 'coll', '=""'; lowers '>', '="/', 'icon', '="../', '="../../', '>{', 'Collection', 'Template'
- promoted: 'imab', ' BGCOLOR', ' suppressed', ' suppress', 'ünd', ' surrounding', 'jä', ' scroll'
- specific promoted: '!.', ')!', '!).', '!', 'alion', 'ages', '__.', ' scroll'
- specific demoted: ' quo', 'gm', 'quet', 'WISE', 'Meet', 'iste', 'gay', 'integral'
- `                                <div class="form-group row">⏎                                    <label for="doc.content" class="[[col]]-sm-3` → '-', help 0.0
- `ul>⏎</li>⏎</ul>⏎</div>⏎<div class="details">⏎<ul[[ class]]="blockList">⏎` → '="', help 0.0
- `</a></li>⏎</ul>⏎</div>⏎<div class="subNav">⏎<ul[[ class]]="navList"` → '="', help 0.0

- pair 0-1: no use found; score -0.01, specific cosine -0.006, raw cosine 0.443

## Neuron 2443: v1 mixed from the start, v2 mixed from the start, v3 no use found (facets 0-1)

Facet-typical activation 1.728; patched into random text at that level, its median help is -0.0, KL 0.000426, sharpen cosine 0.024.

**Facet 0, legal: 'to' in rulings** (40 examples, 7 from v1): harmless, median help 0.0003 (p = 0.8995), later predictions 0.0025 (harmless); reliability 0.785, generic share 0.4

- probability: median KL 0.00042, moved 0.008615, sharpen cosine -0.024; raises ' this', ' it', ' we', ' warrant', ' permit', ' a', ' justify', ' Accordingly'; lowers ' the', ' be', ' require', ' Plaintiff', ' See', ' reversal', ' dispute', ' show'
- promoted: 'ires', 'rss', 'IRS', 'ources', 'idem', 'xsd', 'authorized', 'iens'
- specific promoted: 'ources', 'rfloor', 'obb', 'iger', 'documentclass', 'idem', '======', 'supplementary'
- specific demoted: 'ј', ' yourselves', 'auge', 'gregation', 'LY', ' Yet', ' eye', 'yet'
- `th Cir. 1983). Therefore, the court concluded, Mr. Grissom had not⏎⏎advanced sufficient factual allegations[[ to]] show an Eighth Amendment` → ' show', help -0.118
- ` Benjamin Daniel Kennedy, III's Combined Original Motion to Dismiss Claims for Lack of Subject-Matter Jurisdiction and Failure to[[ State]] Claims Upon Which Relief` → ' Claims', help 0.142
- ` does not appear to us that it was clearly erroneous for the district court and magistrate to determine that the facts are such as[[ to]] warrant detention of Cumm` → ' warrant', help 0.013

**Facet 1, shell: heredoc/redirect** (14 examples, 6 from v1): harmless, median help 0.0121 (p = 0.4631), later predictions 0.0029 (harmless); reliability 0.609, generic share 0.377

- probability: median KL 0.001371, moved 0.015787, sharpen cosine 0.049; raises ' E', ' ng', 'Char', '32', "'", '(', ' END', 'dev'; lowers 'EOF', 'Lower', 'ak', '($_', ' (!', ' display', 'END', 'CMD'
- promoted: 'irus', 'olip', 'ßen', 'ー�', '�', ' contig', 'isher', 'heres'
- specific promoted: 'obuf', 'olip', 'ucht', 'atio', 'criptor', 'othal', 'RUPT', ' ky'
- specific demoted: 'EDIT', 'odel', 'ro', 'section', 'iframe', 'boxes', 'tik', 'unden'
- `              -e "s/\//$ngx_regex_dirsep/g"`⏎+⏎+    cat[[ <<]] END                                                >>` → ' END', help 0.138
- `a-zA-Z_\-]*buildroot/share/fonts|buildroot/share/fonts|' 2[[>/]]dev/null⏎` → 'dev', help 0.009
- ` stderr"⏎ fi⏎ ⏎+if [ $NGX_DTRACE = YES ]; then⏎+    cat[[ <<]] END⏎+  ` → ' END', help 0.107

- pair 0-1: no use found; score -0.063, specific cosine -0.043, raw cosine 0.361

## Neuron 1391: v1 mixed from the start, v2 mixed from the start, v3 no use found (facets 0-2)

Facet-typical activation 1.348; patched into random text at that level, its median help is -0.0, KL 0.000305, sharpen cosine -0.015.

**Facet 0, legal: S.Ct. pages** (40 examples, 6 from v1): harmless, median help -0.0008 (p = 0.0241), later predictions -0.0004 (harmless); reliability 0.74, generic share 0.549

- probability: median KL 0.000261, moved 0.007727, sharpen cosine -0.08; raises ' (', '    ', ' defendant', ' .', ' harassment', ' government', ' request', '8'; lowers ',', ' the', '.', '                        ', ' to', '     ', ' plaintiff', ' United'
- promoted: 'ises', 'elijk', 'chos', 'hores', 'aries', 'ment', 'mas', 'ters'
- specific promoted: 'empre', 'ionale', 'ises', 'iale', 'FREE', 'woke', 'STAND', 'icular'
- specific demoted: '.]', ' Search', 'esti', '{(\\', ' Fin', ".'", '{}\\', ' U'
- `. Warden, Maryland Penitentiary, 372 U.S. 776, 83 S.Ct. 110[[3]], 10 L.` → ',', help 0.001
- `4th Cir. 1967), cert. denied, 390 U.S. 905, 88 S.Ct. 8[[18]], 19 L.` → ',', help 0.0
- `. at 570, 127 S.Ct. 1955). The Court explained that "the pleading standard Rule 8 announces does not[[ require]] `detailed factual` → ' `', help -0.044

**Facet 1, sports: 'defeated by/to'** (17 examples, 5 from v1): harmless, median help -0.002 (p = 0.6777), later predictions 0.0036 (harmless); reliability 0.545, generic share 0.376

- probability: median KL 0.00038, moved 0.010977, sharpen cosine -0.02; raises ' a', 'ensively', ' foul', ' W', ' for', ' V', ' players', ' Mar'; lowers ' the', ' game', ' it', ' Manchester', '-', ' Arsenal', ' Mitt', ' Liverpool'
- promoted: 'ment', 'gery', 'uate', 'uated', 'uates', 'ufact', 'ce', 'uality'
- specific promoted: 'uate', ' Authority', 'gery', 'uction', 'ivation', 'ITERATION', 'ufact', 'sale'
- specific demoted: 'roke', 'RL', 'enas', 'iets', 'PAT', 'enos', 'en', 'alam'
- `ase at the 10:08 mark before Calgary’s Dan Taylor provided yet another equalizer less than 30 seconds later to[[ force]] overtime.⏎⏎` → ' overtime', help -0.002
- ` Dubs: ‘They've done it in big moments'⏎⏎Defending champions set for 2015 WCF rematch[[ with]] Rockets⏎⏎` → ' Rock', help 0.028
- ` with one win, two losses and a tie. They should be fresh, having not played since a 1-0 defeat[[ to]] the Seattle Sounders` → ' the', help -0.009

**Facet 2, travel: '(GOOGLE MAP)'** (40 examples, 5 from v1): harmless, median help 0.0 (p = 0.1461), later predictions 0.0 (harmless); reliability 0.979, generic share 0.271

- probability: median KL 0.0, moved 1e-06, sharpen cosine -0.041; raises ')', 'ading', 'ript', 'ard', ' v', ');', 'icki', 'm'; lowers ';', 'agons', 'rolling', 'on', ' 45', 'icker', 'arg', 'aid'
- promoted: 'edom', 'itas', 'iles', 'igraphy', 'esp', 'ises', 'ility', 'zing'
- specific promoted: 'igraphy', 'aryng', 'zin', 'eters', 'ila', 'iles', 'eth', 'zing'
- specific demoted: 'vere', '\'">', '\'"', '"|', '       ', '"-', '\'?"', '"?"'
- ` it's cocktails, candles and Belgian and Japanese beers.⏎⏎Hall for CornwallTHEATRE⏎⏎(  [[G]]OOGLE MAP ) ;` → 'OOGLE', help -0.0
- `es, cakes, homemade soups and sandwiches.⏎⏎Waterfront Fishouse RestaurantSEAFOOD⏎⏎(  [[G]]OOGLE MAP ) ;` → 'OOGLE', help -0.0
- ` classics here on summer evenings, along with comedy and concerts.⏎⏎ZSL London ZooZOO⏎⏎(  [[G]]OOGLE MAP ) ;` → 'OOGLE', help -0.0

- pair 0-1: no use found; score 0.027, specific cosine 0.017, raw cosine 0.464
- pair 0-2: no use found; score 0.15, specific cosine 0.128, raw cosine 0.459
- pair 1-2: no use found; score 0.042, specific cosine 0.03, raw cosine 0.34

## Neuron 61: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.242; patched into random text at that level, its median help is -0.0, KL 0.000253, sharpen cosine 0.01.

**Facet 0, facet 0** (24 examples, 5 from v1): harmless, median help 0.0006 (p = 0.8774), later predictions 0.0023 (harmless); reliability 0.52, generic share 0.634

- probability: median KL 0.000308, moved 0.008885, sharpen cosine -0.02; raises ' it', ' unconscious', ' one', ' prayer', ' article', ' with', ' and', ' have'; lowers ' too', ' for', ' be', ' I', ' he', ' are', ' in', ' consciousness'
- promoted: 'arson', 'bum', '@', 'riz', 'atted', 'rien', 'nered', 'ku'
- specific promoted: 'ttp', 'atted', 'acia', 'asi', 'ae', 'iana', 'ont', 'ACE'
- specific demoted: ' unrestricted', ' independent', '�', ' list', ' unrelated', ' forgot', ' other', ' freely'
- ` can anticipate it correctly and live through it.⏎⏎Meanwhile, in a curious and unanticipated way, we[[ really]] do welcome the birth` → ' do', help 0.011
- ` rethink the ontology through negative theology and a mystical way. Since Heidgger the reality is seen as a gift from[[ which]] the philosophy would indicate` → ' the', help -0.004
- `ures daily to see whether these things were so.⏎⏎We have seen that the Scriptures have spoken: we must[[ not]] do any action while` → ' do', help -0.009

**Facet 1, facet 1** (26 examples, 5 from v1): harmless, median help 0.0008 (p = 0.9007), later predictions -0.0005 (harmless); reliability 0.588, generic share 0.562

- probability: median KL 0.000295, moved 0.00785, sharpen cosine -0.031; raises '\n\t\t\t\t', ' don', ' S', ' super', ' V', ' bool', '\n', ' no'; lowers '\n\t\t\t', 'return', 'unsigned', ' struct', ' want', '\n\t\t', 'j', ' do'
- promoted: '///', '.*]{},', '@{', ' only', 'aska', ' shower', 'ishers', 'apper'
- specific promoted: 'orin', 'avery', '■', ' shower', ' only', 'track', 'ishers', 'IE'
- specific demoted: '"/', ')"', '"[', '"', '"(', ')', 'ako', '("'
- `handle     Handler⏎	cancel     Canceler⏎	log        Logger⏎	stream     Stream⏎	done       [[chan]] struct{}⏎	` → ' struct', help -0.011
- `32_t helper_evaluate_flags_mulu(uint32_t ccs, uint32_t res[[,]] uint32_t` → ' uint', help 0.003
- ` 1)) - 1))⏎⏎void helper_movl_sreg_reg (uint32_t sreg[[,]] uint32_t` → ' uint', help 0.004

- pair 0-1: no use found; score -0.081, specific cosine -0.045, raw cosine 0.579

## Neuron 4010: v1 packing-like, v2 recursive generalization, v3 no use found (facets 0-1)

Facet-typical activation 1.612; patched into random text at that level, its median help is -0.0, KL 0.000435, sharpen cosine -0.016.

**Facet 0, facet 0** (40 examples, 8 from v1): harmless, median help 0.0 (p = 0.6085), later predictions -0.0 (harmless); reliability 0.752, generic share 0.51

- probability: median KL 0.000299, moved 0.009441, sharpen cosine -0.003; raises 'e', ' no', ' leak', 'Ill', ' agar', ' \\~', ' rug', '1'; lowers 'al', ' poor', ' No', ' 25', ' caps', '\xa0', ' protein', ' \\#'
- promoted: 'auss', '*,', 'im', 'bris', 'agem', 'olis', '[/', ' Material'
- specific promoted: 'elike', 'shape', 'kel', 'derived', 'ques', 'vector', ' text', ' nick'
- specific demoted: 'unnumbered', '\r\n ', '>>>>', '.__', '\n ', '‑', '/-/', ' _________________________________'
- `. Reprinted with permission from ref. \[[@B77-jfb-09-00008]\]. Copyright (2016[[)]] Nature Publishing Group.](` → ' Nature', help 0.038
- `01, A11:01, and C03:04, which account for 19.5%, 5.6%,[[ and]] 4.7%` → ' 4', help -0.007
- `CP--Cellular Processes, EIP--Environmental Information Processing, GIP--Genetic Information Processing, M[[--]]Metabolism, OS` → 'Met', help 0.036

**Facet 1, facet 1** (40 examples, 6 from v1): harmless, median help -0.0002 (p = 0.1119), later predictions 0.0003 (harmless); reliability 0.57, generic share 0.397

- probability: median KL 0.000429, moved 0.011148, sharpen cosine 0.009; raises 'R', ' Van', '.', ' Jean', 'W', ',', ' At', ' David'; lowers 'Court', ' Civil', ' (', ' for', ' sub', '\n', ' Circuit', ' G'
- promoted: 'aland', 'EN', 'umn', '**_', '_**', 'en', 'alar', 'til'
- specific promoted: 'EN', '_**', 'Pa', 'Jac', '**_', '"""', '�', 'Gary'
- specific demoted: 'rogens', 'inib', 'ifying', 'naments', 'batim', 'rification', 'lining', 'ikov'
- ` GONZALES III⏎                                           V.⏎                               THE STATE OF TEXAS⏎⏎⏎                                 On Appeal from[[ the]]⏎                    105th` → '\n', help 0.015
- ` jurisdiction,” the resulting judgment is voidable rather than void.⏎⏎Dunbar v. State, 136 Ohio St.[[3]]d 181, 2013` → 'd', help -0.0
- `COND DISTRICT⏎⏎_________________________________________________________________⏎⏎⏎⏎THE PEOPLE OF THE STATE       )  Appeal from[[ the]] Circuit Court⏎⏎` → ' Circuit', help -0.019

- pair 0-1: no use found; score 0.096, specific cosine 0.063, raw cosine 0.484

## Neuron 682: v1 mixed from the start, v2 mixed from the start, v3 no use found (facets 0-1)

Facet-typical activation 1.39; patched into random text at that level, its median help is -0.0001, KL 0.000418, sharpen cosine 0.013.

**Facet 0, facet 0** (40 examples, 8 from v1): harmless, median help 0.0016 (p = 0.7146), later predictions 0.0027 (harmless); reliability 0.833, generic share 0.411

- probability: median KL 0.00044, moved 0.011164, sharpen cosine 0.007; raises ' I', ' that', ' fault', ' what', ' her', "'d", ' he', ' would'; lowers ' was', ' to', ' the', ' an', ' it', ' voice', ' hit', ' something'
- promoted: ' temptation', 'ţi', 'esk', ' recognizing', 'aware', 'es', 'vision', 'uum'
- specific promoted: 'Accessor', ' glimpse', ' temptation', ' glance', ' query', 'query', ' allowing', 'than'
- specific demoted: ' Excellence', ' PS', ' Handbook', 'coal', ' Pract', ' Phys', 'BN', 'RSP'
- `. You became the sweetest, gentlest dog, who was always wary around her. You always eyed me like you[[ knew]] you'd done wrong` → ' you', help 0.079
- ` at me. I felt that she was talking to Father when she was seized in this way, and perhaps I, without[[ realizing]] it, had become` → ' it', help 0.008
- `to move. A moment later he caught a step at his side, and opened his⏎eyes, thinking that Ed[[ had]] returned.⏎⏎` → ' returned', help 0.005

**Facet 1, facet 1** (40 examples, 6 from v1): harmless, median help 0.0014 (p = 0.3472), later predictions 0.0018 (harmless); reliability 0.974, generic share 0.059

- probability: median KL 0.000774, moved 0.013322, sharpen cosine 0.01; raises '.', '5', '3', '4', '2', '1', '25', '85'; lowers '10', '45', '/', '50', '12', '35', '60', '-'
- promoted: ' 05', ' 06', ' case', 'achy', ' instance', ' 04', ' -', 'lay'
- specific promoted: ' 06', 'achy', ' 05', ' 04', ' 07', 'ellate', ' instance', 'unn'
- specific demoted: '425', '495', '380', '420', '499', '525', '470', '465'
- `MacRae Guest HouseB&B⏎⏎(  MAP   GOOGLE MAP ) ;  %01463-[[24]]3658; joy` → '36', help 0.012
- `-250654; www.boscastlehouse.co.uk; Tintagel Rd; s/d from[[ £]]50/102;` → '50', help -0.028
- `909; www.glendevonguesthouse.co.uk; 8 Truro Rd; s/d from[[ £]]52.50/` → '52', help -0.042

- pair 0-1: no use found; score -0.02, specific cosine -0.018, raw cosine 0.143

## Neuron 1228: v1 packing-like, v2 mixed from the start, v3 no use found (facets 2-3)

Facet-typical activation 1.957; patched into random text at that level, its median help is 0.0001, KL 0.000608, sharpen cosine -0.01.

**Facet 0, facet 0** (40 examples, 7 from v1): harmless, median help 0.0 (p = 0.4358), later predictions 0.0004 (harmless); reliability 0.892, generic share 0.35

- probability: median KL 5.6e-05, moved 0.000601, sharpen cosine 0.019; raises 'rd', ' Cal', ' R', ' for', '81', ' Wash', ' Pa', ' Or'; lowers 'd', ' U', ' Md', ' F', ' Minn', ' to', ' N', ' A'
- promoted: 'ova', ' than', 'OVA', 'than', '----------------------------------------------------------------------', '/−', 'itum', 'gom'
- specific promoted: 'gom', 'ova', '/−', ' clearance', ' TOP', '----------------------------------------', 'dra', ' protection'
- specific demoted: 'ems', 'arp', ' Series', 'inian', 'arker', 'urns', 'ér', 'ad'
- `., 533 U.S. 53, 62, 65 (2001);⏎see also Trimble v. Gordon,[[ 430]] U.S.` → ' U', help -0.024
- `, § 79, p. 3240; see, e.g., Lesser & Son v. Seymour.[[ 35]] Cal.2d` → ' Cal', help 0.002
- `). See 1A Corbin on Contracts § 263 (1963). See generally Kahn v. General Development Corp.,[[ 40]] Del. Ch.` → ' Del', help 0.087

**Facet 1, facet 1** (8 examples, 5 from v1): harmless, median help 0.0031 (p = 0.1953), later predictions 0.0005 (harmless); reliability 0.805, generic share 0.304

- probability: median KL 0.000181, moved 0.004416, sharpen cosine -0.032; raises '{', 'S', ' R', 'L', 'D', ' C', 'M', 'P'; lowers ' {', '}', 'C', ' H', ' A', ' N', ' P', ' F'
- promoted: 'imming', '×', 'ches', 'lement', 'ger', 'xiv', 'yll', ' xx'
- specific promoted: 'xx', '×', 'lement', 'XX', 'eller', 'other', ' xx', 'eler'
- specific demoted: 'raf', '/~', 'bows', 'PO', 'bane', 'akis', 'akan', ' steps'
- `b,\sigma\in {\ensuremath {\mathbb{R}}}$ and $\nu$ is a measure on ${\ensuremath {\mathbb[[{]]R}}}$ satisfying $$\` → 'R', help 0.001
- `h_i\in L_2({\ensuremath {\mathbb{R}}})$ and $\sigma\in{\ensuremath {\mathbb[[{]]R}}}_{>0` → 'R', help -0.004
- `possibly infinite) Lévy measure $\nu_0$ concentrated on a possibly infinite interval $I\subseteq{\ensuremath {\[[mathbb]]{R}}}$, admitting` → '{', help 0.036

**Facet 2, facet 2** (13 examples, 5 from v1): harmless, median help -0.0004 (p = 0.7354), later predictions -0.0011 (harmless); reliability 0.766, generic share 0.221

- probability: median KL 0.000623, moved 0.008406, sharpen cosine -0.064; raises '\xa0', 'l', ' Lett', ' Rep', 'th', 'held', ' C', ' Supp'; lowers ' Rev', 'wal', ' U', 'we', 'berg', 'by', 'b', 'w'
- promoted: 'onel', 'ani', 'ALL', 'gart', 'avian', 'act', 'vis', 'hex'
- specific promoted: 'vis', 'án', 'get', 'are', 'anchor', 'rett', 'rades', 'serial'
- specific demoted: 'FFIR', ')[@', ')",', ' dim', ' fever', '$^{-', ")',", 'UX'
- `.*]{} [**B 94**]{}, 224418 (2016). Mazurenko, V. V., et al. [*Phys[[.]] Rev.*]{} [**B` → ' Rev', help -0.003
- ` Rev. B* [**69**]{} 161402(R); Xue Y and Ratner M A 2004 *Phys[[.]] Rev. B*` → ' Rev', help -0.001
- ` 2000 *Phys. Rev. B* [**62**]{} 5238 Xue Y and Ratner M A 2003 *Phys[[.]] Rev. ` → '\xa0', help 0.158

**Facet 3, facet 3** (11 examples, 4 from v1): harmless, median help -0.0 (p = 0.5771), later predictions 0.0 (harmless); reliability 0.775, generic share 0.35

- probability: median KL 4.4e-05, moved 0.000465, sharpen cosine 0.01; raises '  ', ' C', ' F', ' or', ' be', ' also', ' Wash', ' have'; lowers '\n\n', ' In', '3', ' The', ' know', ' U', ' still', ','
- promoted: ' than', 'ova', 'ana', '----------------------------------------------------------------------', 'OVA', 'ahoma', 'dra', 'uest'
- specific promoted: 'dra', 'infall', 'ailable', ' win', ' Vegas', 'alc', ' chall', ' dra'
- specific demoted: 'j', 'PI', 'paragraph', '_."', 'reng', 'omon', ' 355', 'Princ'
- ` meaning of the section so unambiguous or straightforward as to override the command of § 152(b)." See Iowa Utilities,[[ 119]] S. Ct.` → ' S', help -0.0
- ` FCC failed to "articulate[] a rational relationship between the facts found and the choice made . . . ." Harris,[[ 19]] F.3d` → ' F', help 0.001
- ` Because the City's ordinance contravenes this policy choice, it must give way. See Seattle Newspaper-Web,[[ 24]] Wash.App.` → ' Wash', help 0.001

**Facet 4, facet 4** (19 examples, 5 from v1): harmless, median help 0.0 (p = 0.5678), later predictions 0.0001 (harmless); reliability 0.703, generic share 0.273

- probability: median KL 0.000321, moved 0.006168, sharpen cosine 0.027; raises 'Writer', 'stderr', 'path', '#', 'St', 'error', '    ', ' *'; lowers 'argv', '\n', 'err', 'Std', 'Write', 'Reader', 'set', 'exit'
- promoted: '@', ' @', 'yler', '*^*', '*', 'thood', ' dest', ' *'
- specific promoted: ' @', ' q', ' Q', ' ***', '@', ' i', ' 70', '[\\*](#'
- specific demoted: 'planes', 'lemen', 'NOW', 'anyahu', 'suits', 'lish', ' Lebanon', 'gether'
- `acc_badargs.py⏎#⏎# Rules with wrong # args⏎# -----------------------------------------------------------------------------⏎import sys⏎sys[[.]]tracebacklimit =` → 'trace', help -0.027
- `// Prints to the standard logger if debug mode is enabled. Arguments⏎// are handled in the manner of fmt[[.]]Println.⏎` → 'Print', help 0.002
- ` back to os.Stderr.⏎src/pkg/log/log.go⏎var std = New(os[[.]]Stderr, "",` → 'St', help 0.018

- pair 0-1: too few examples; score 0.106, specific cosine 0.09, raw cosine 0.387
- pair 0-2: no use found; score 0.235, specific cosine 0.194, raw cosine 0.416
- pair 0-3: no use found; score 0.813, specific cosine 0.676, raw cosine 0.79
- pair 0-4: no use found; score 0.158, specific cosine 0.125, raw cosine 0.395
- pair 1-2: too few examples; score 0.08, specific cosine 0.063, raw cosine 0.306
- pair 1-3: too few examples; score 0.006, specific cosine 0.004, raw cosine 0.329
- pair 1-4: too few examples; score 0.163, specific cosine 0.122, raw cosine 0.375
- pair 2-3: no use found; score 0.274, specific cosine 0.211, raw cosine 0.429
- pair 2-4: no use found; score -0.05, specific cosine -0.036, raw cosine 0.219
- pair 3-4: no use found; score 0.017, specific cosine 0.013, raw cosine 0.318

## Neuron 1345: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.307; patched into random text at that level, its median help is -0.0, KL 0.000371, sharpen cosine -0.011.

**Facet 0, fiction verbs** (40 examples, 11 from v1): harmless, median help -0.0029 (p = 0.536), later predictions -0.0001 (harmless); reliability 0.842, generic share 0.464

- probability: median KL 0.000468, moved 0.009916, sharpen cosine -0.033; raises ' to', '.', ' with', ' her', ' nothing', ' that', ' as', ' this'; lowers ',', ' about', ' of', ' them', ' on', ' back', ' thought', ' over'
- promoted: 'lands', ' hereafter', 'wise', ' therefrom', 'lier', ' tomorrow', 'other', 'heres'
- specific promoted: 'ия', 'never', 'αι', 'izes', 'je', 'aneously', 'bin', 'izable'
- specific demoted: 'WAIT', 'UP', 'otts', '+.', ' collected', 'arget', ' attent', 'ICE'
- ` mean by last chance" she was a little nervous for some reason, "but yes I think I..." before she could[[ finish]] a flash came from` → ' a', help -0.034
- ` burning needles. One way or another, she must write about your clothes."⏎⏎The queen broke off, suddenly distracted[[ by]] her makeup. She` → ' her', help 0.056
- ` room, and with him had come pure evil.⏎⏎That was just the way things were in life, she told[[ herself]]. Bad things happened` → '.', help -0.01

**Facet 1, math expressions** (40 examples, 5 from v1): harmless, median help -0.0 (p = 0.7648), later predictions -0.0 (harmless); reliability 0.895, generic share 0.34

- probability: median KL 1e-06, moved 3.4e-05, sharpen cosine -0.075; raises 'd', ' +', '0', 'v', '_', 'Differentiate', 'Find', ',\\'; lowers ',', 'What', ' -', '4', 'a', ' x', 'Suppose', 'k'
- promoted: ' this', ' notice', ' pro', 'opin', '[])', ' dat', ' these', 'ruit'
- specific promoted: 'CR', ' this', ' DE', ' state', ' CR', ' so', ' de', ','
- specific demoted: ' pesso', 'fro', '~-', 'rapeut', '\'?"', 'queous', '+-+-', ' CONTRIBUTORS'
- `r**2 + 6*r + 5. Let b(n) = -6*n**2 - 5[[*]]n - 4.` → 'n', help -0.0
- `c) = -13*c**3 + 4*c**2 - 3*c. Let h(f[[)]] = -5*` → ' =', help 0.0
- `. Let d(x) = -19*x**2 + 7*x - 9. Let z(y[[)]] = -d(` → ' =', help 0.0

- pair 0-1: no use found; score -0.018, specific cosine -0.015, raw cosine 0.388

## Neuron 3370: v1 packing-like, v2 packing-like, v3 no use found (facets 0-2)

Facet-typical activation 1.365; patched into random text at that level, its median help is 0.0, KL 0.000335, sharpen cosine 0.049.

**Facet 0, dialogue: 'Sure'** (34 examples, 7 from v1): harmless, median help 0.0015 (p = 0.5429), later predictions -0.0028 (harmless); reliability 0.892, generic share 0.459

- probability: median KL 0.000193, moved 0.00649, sharpen cosine 0.089; raises ' thing', ',', ' yes', ' and', ' of', '..."', ' you', ' but'; lowers ' that', 'ly', '?"', ' it', '.', '."', ' L', ',"'
- promoted: 'onto', ']{})', 'emis', ' LINEAR', '$)', 'ename', 'mons', 'isc'
- specific promoted: 'of', 'onto', 'mys', 'name', 'elli', ' lapse', ' name', '\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0'
- specific demoted: 'izers', ' ranges', 'VERTIS', ' Contributions', ' Should', 'Ns', ' Methods', ' lett'
- `." "The kiss was everything." "But it wasn't enough." "Darling... you buy me drink?" "[[Sure]]." "What would` → '."', help 0.007
- ` the show?" "But I can't." "I have paints." "Can I talk to you, please?" "[[Sure]], let's go` → ',', help 0.003
- ` become a very good screenwriter." "Thank you, Tina." "You should ask her to read your treatment." "[[Sure]]." "Sure,` → '."', help 0.018

**Facet 1, dialogue: 'Sure' (2)** (40 examples, 6 from v1): harmless, median help 0.0016 (p = 0.1499), later predictions 0.0022 (harmless); reliability 0.861, generic share 0.512

- probability: median KL 0.000241, moved 0.008537, sharpen cosine 0.026; raises ',', ' did', ' he', ' side', ' you', 'ine', ' let', 'We'; lowers ' do', 'ines', ' that', '?"', ' was', ' enough', ' I', ' then'
- promoted: 'uria', 'iani', 'ename', ' NAME', 'eni', 'omen', '$)', 'ora'
- specific promoted: 'ora', ' NAME', 'orus', ' name', 'aran', 'eni', 'of', 'asso'
- specific demoted: ' lett', '[[', ' Means', 'taire', 'means', 'eeper', 'rends', ' Gets'
- `. You believe me right?"⏎⏎I bumped my arm against his and gave him my most innocent smile. "[[Sure]], I believe you` → ',', help 0.033
- ` briefcase." "But I couldn't swear about brown." " He came out of the hotel?" " Yeah." "[[Sure]], that figures."` → ',', help 0.092
- ` at my soaked clothes, I debated, but it was a nice night and I really wanted to go.⏎⏎"[[Sure]], I can't` → ',', help 0.012

**Facet 2, Japanese/Chinese particles** (17 examples, 5 from v1): harmless, median help 0.0057 (p = 0.89), later predictions -0.001 (harmless); reliability 0.806, generic share 0.465

- probability: median KL 0.000791, moved 0.016411, sharpen cosine 0.067; raises '大', '多', 'つ', '�', 'あ', '�', '�', '常'; lowers '好', '�', 'ある', 'する', 'の', '取', '\u3000', '�'
- promoted: 'if', 'enary', 'idi', 'panic', 'ner', 'lines', 'ces', 'iani'
- specific promoted: 'if', 'panic', 'each', 'z', 'stand', 'er', 'idi', 'ming'
- specific demoted: 'andum', 'に', 'ikip', 'Pane', 'です', 'ゃ', 'Components', 'ensely'
- `�ートカットのバレーボール選手。ポジション、身長など[[も]]木村` → '�', help -0.025
- `nid=27)  ⏎[[PDF] Google 官方的 AngularJS 应用结构[[最]]佳实�` → '�', help 0.045
- `�果・効能、日本におけるCBDに関する法律や制度ま[[で]]、徹�` → '、', help 0.034

- pair 0-1: no use found; score 0.906, specific cosine 0.794, raw cosine 0.893
- pair 0-2: no use found; score 0.168, specific cosine 0.142, raw cosine 0.538
- pair 1-2: no use found; score 0.077, specific cosine 0.064, raw cosine 0.521

## Neuron 2019: v1 packing-like, v2 packing-like, v3 no use found (facets 1-2)

Facet-typical activation 1.45; patched into random text at that level, its median help is -0.0, KL 0.000366, sharpen cosine 0.015.

**Facet 0, legal: '(holding'** (40 examples, 6 from v1): harmless, median help 0.0053 (p = 0.2016), later predictions 0.0018 (harmless); reliability 0.876, generic share 0.37

- probability: median KL 0.001297, moved 0.020347, sharpen cosine 0.033; raises 'emphasis', 'en', 'per', 'CC', '2001', ' 1', 'up', 'affirm'; lowers 'citing', 'stating', '[', 'noting', '1979', 'finding', '2002', '1980'
- promoted: 'ikip', 'onymous', ' infinitely', 'oney', 'feit', 'nier', ' lips', 'onym'
- specific promoted: 'pread', 'ten', 'feit', '!)', ' infinitely', ' stroke', ' alike', ' insufficient'
- specific demoted: '\u2002', '---|---', 'Pars', 'Hal', 'odb', 'crystall', 'itas', 'ία'
- ` 590, 94 S.Ct. 2464, 2469, 41 L.Ed.2d 325 (1974)[[ (]]plurality opinion` → 'pl', help 0.066
- ` S. Calif., 206 Cal.App.3d 380, 253 Cal.Rptr. 541, 542 (1988)[[ (]]covered), cert.` → 'covered', help 0.206
- ` (22 NW2d 450, 455).⏎Cf. Robertson v. M/S Sanyo Maru[[ (]]CA5, 1967` → 'CA', help 0.123

**Facet 1, legal: parentheticals** (31 examples, 5 from v1): harmless, median help 0.0 (p = 0.5814), later predictions 0.0001 (harmless); reliability 0.883, generic share 0.406

- probability: median KL 0.00059, moved 0.01391, sharpen cosine 0.025; raises 'quoting', 'internal', 'trial', 'per', 'HD', 'en', 'emphasis', 'pl'; lowers 'citing', '5', 'stating', 'con', 'citation', 'noting', 'finding', 'the'
- promoted: 'ikip', ' lips', 'imal', ' encore', ' infinitely', 'term', 'scriber', 'onymous'
- specific promoted: 'lik', ' lifes', 'imal', 'pread', 'yond', 'BASE', 'gan', ' slam'
- specific demoted: 'ISHED', 'orel', 'ished', 'Hal', ' Cron', 'Pars', ' Olson', ' Cir'
- ` logical end of this line in Sumner v. Shuman, 483 U.S. 66, 85 (1987)[[ (]]holding mandatory death penalty` → 'holding', help 0.012
- ` DeBella v. People,⏎⏎  233 P.3d 664, 667 (Colo. 2010)[[ (]]failure to exercise discretion` → 'failure', help 0.049
- `have   been    anticipated       by   the    original⏎⏎tortfeasor.”             Id.        [[(]]internal        quotation` → 'internal', help 0.008

**Facet 2, materials: timber/steel** (40 examples, 5 from v1): harmless, median help -0.0023 (p = 0.1461), later predictions 0.0007 (harmless); reliability 0.806, generic share 0.491

- probability: median KL 0.000722, moved 0.015827, sharpen cosine -0.006; raises 'and', '-', 'burn', 'degree', 'frame', ' TV', ' display', ' gr'; lowers 'flo', 'cl', 'pan', 'wall', 'es', ' are', 'fired', '.'
- promoted: 'egen', 'ikip', 'even', 'letters', 'DEP', 'Shift', 'Lear', 'handle'
- specific promoted: 'letters', 'letter', 'Letter', 'Delegate', 'arrows', ' forfeiture', 'Speed', 'egen'
- specific demoted: 'rapy', 'ivable', 'ri', 'ran', 'riz', 'ovo', 'itic', 'opter'
- ` Features include handcrafted from pine wood, 3 center shelves for your hardware, 2 side cabinets and fitted with forged iron[[ metal]] handles to add a` → ' handles', help 0.066
- `plays on fishing, shipbuilding, coastal defences and Regency-era tourism set in an intricately carved, timber[[-]]framed, 16` → 'fr', help -0.016
- `p i W s)⏎⏎Stone walls, worn flagstones, secluded gardens and one room with an extraordinary timber[[-]]vaulted ceiling` → 'v', help 0.05

- pair 0-1: no use found; score 0.959, specific cosine 0.843, raw cosine 0.903
- pair 0-2: no use found; score -0.075, specific cosine -0.063, raw cosine 0.391
- pair 1-2: no use found; score -0.061, specific cosine -0.052, raw cosine 0.418

## Neuron 1814: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.712; patched into random text at that level, its median help is -0.0, KL 0.000441, sharpen cosine 0.007.

**Facet 0, math problems: 'What is/Let'** (40 examples, 12 from v1): harmless, median help -0.0002 (p = 0.0185), later predictions -0.0003 (harmless); reliability 0.917, generic share 0.249

- probability: median KL 7.9e-05, moved 0.004728, sharpen cosine 0.027; raises ' o', ' t', ' z', ' i', ' f', ' w', ' h', ' l'; lowers ' s', ' d', ' 3', ' u', ' 2', ' 4', ' 5', ' a'
- promoted: ' 0', ' ¼', 'uty', ' \\%', '**.', ' phr', 'un', 'unt'
- specific promoted: ' 0', ' "\\', 'unt', ' ¼', ' =\\', ' -*-', ' =', ' phr'
- specific demoted: 'iary', 'oth', 'ettes', 'ette', 'orts', 'orting', 'generic', 'orters'
- `b**3 + 2*b**2 + 9*b - 13. Determine x(-3).⏎5⏎[[Let]] g(w)` → ' g', help -0.015
- ` = -x - 4. What are the prime factors of b(x)?⏎2, 23⏎Suppose -i[[ -]] 3*i =` → ' 3', help -0.032
- `8627?⏎-9*n**3 - 51*n**2 - 95*n + 5⏎What[[ is]] the h'th` → ' the', help -0.0

**Facet 1, fiction: 'I thought'** (40 examples, 9 from v1): harmless, median help 0.0072 (p = 0.0034), later predictions -0.0002 (harmless); reliability 0.799, generic share 0.502

- probability: median KL 0.00065, moved 0.013315, sharpen cosine -0.01; raises ' up', '?"', ' Mr', ',"', '."', ' exams', ' Cant', ' on'; lowers ' "', ' it', ' I', ' expecting', " '", '\n', ',', ' to'
- promoted: 'ause', 'ends', '%).', '%)', ' forever', 'ang', 'abl', 'leep'
- specific promoted: 'please', ' literal', 'emat', 'ang', '.’', ")',", '_)', 'we'
- specific demoted: 'pntd', 'pcbi', 'LLOW', ' Same', 'Same', 'iquity', 'eft', 'autre'
- ` eyed⏎Denis.⏎⏎“Well, for the love of Mike!” he ejaculated slowly. “Th[[ought]] you had⏎v` → ' you', help -0.026
- ` you did that." She paused long enough to put her hands on my shoulders and look me in the eyes. "I[[ thought]] I was going to` → ' I', help 0.001
- ` time, I'm going to grind their CD under a spiky heel."⏎⏎Daniel was wounded. "I[[ thought]] you liked them."` → ' you', help 0.021

- pair 0-1: no use found; score 0.065, specific cosine 0.055, raw cosine 0.387

## Neuron 3915: v1 packing-like, v2 packing-like, v3 no use found (facets 0-3)

Facet-typical activation 2.301; patched into random text at that level, its median help is 0.0, KL 0.001092, sharpen cosine -0.01.

**Facet 0, dialogue: 'Yeah'** (40 examples, 9 from v1): harmless, median help -0.0014 (p = 0.9417), later predictions -0.0007 (harmless); reliability 0.879, generic share 0.353

- probability: median KL 0.002258, moved 0.020011, sharpen cosine -0.089; raises '."', ',"', '.', 'Yes', ',”', ' think', ' but', '!”'; lowers ',', ' do', '?"', ' she', ' I', ' thing', ' as', ' guess'
- promoted: '.”', '.“', ',”', '.’', 'yrus', '.\\"', '\u202f', '″'
- specific promoted: '\u202f', '.“', '.”', '“', '----------------------------------------------------------------------------------------------------------------', '.\\"', ',”', '@"'
- specific demoted: ' careful', 'otes', 'olved', ' verbs', 'pload', 'getText', 'ots', 'ota'
- ` a newspaper, a movie." "She's convinced she's encountered it already." " That's a real condition?" "[[ Yeah]], it's called` → ',', help -0.093
- ` he had a briefcase." "But I couldn't swear about brown." " He came out of the hotel?" "[[ Yeah]]." "Sure,` → '."', help 0.045
- ` place," Jamie said with a nod. "It's a pleasure to meet you, Miss McCoy."⏎⏎"Indeed[[ it]] is," Moses added` → ' is', help 0.002

**Facet 1, dialogue: 'Yeah' (2)** (25 examples, 6 from v1): harmless, median help 0.0027 (p = 0.9158), later predictions -0.0002 (harmless); reliability 0.818, generic share 0.35

- probability: median KL 0.001524, moved 0.022317, sharpen cosine -0.087; raises '."', ',"', '.', ' as', 'Yes', 'Yeah', ' but', ' know'; lowers ' I', ',', ' all', ' the', '?"', ' your', 'No', ' yours'
- promoted: 'aq', '--------------------------------------------------------------------------------------------------------------------------------', '@"', 'ostic', '----------------------------------------------------------------------------------------------------------------', ' Apparently', 'arcer', 'But'
- specific promoted: '@"', '\u202f', ' &', '----------------------------------------------------------------------------------------------------------------', 'And', '--------------------------------------------------------------------------------------------------------------------------------', ':\\', '&'
- specific demoted: ' careful', 'olved', 'hin', '>]', 'gated', ' pardon', ' forgiven', ' theirs'
- ` him and he dodged it expertly.⏎⏎"I missed," I said, pretending frustration.⏎⏎"[[Yeah]], you just can` → ',', help -0.016
- ` well." "He must definitely be doing well." "If he isn't, we will be bothered, right?" "[[Yeah]]." "Welcome."` → '."', help 0.067
- ` tries," Jamie said. "Sometimes he has to try a bunch of times."⏎⏎"You're right, of[[ course]]." Moses took a` → '."', help -0.044

**Facet 2, code: aligned constants** (17 examples, 6 from v1): harmless, median help -0.0 (p = 0.6112), later predictions 0.0 (harmless); reliability 0.806, generic share 0.266

- probability: median KL 8.6e-05, moved 0.000488, sharpen cosine -0.002; raises '    ', '  ', 'in', '     ', 'ti', 'art', 'ri', 'g'; lowers '      ', ' =', 't', ' V', '        ', 'tu', ' {', 'ov'
- promoted: ' ],', '+}', 'loop', 'ishly', ' \\}', ' </', 'enough', ' ]'
- specific promoted: ' ],', ' </', '\r\n\r\n ', 'reth', ' improvement', 'ishing', ' ]', '"}](#'
- specific demoted: '.', '?', '?**', '.|', '.**', '.*', '*.', '**.'
- `_PROCESS                      = 0x0⏎	PRIO_USER                         = 0x2⏎	PROT_EXEC[[                        ]] = 0x4` → ' =', help -0.0
- `   = 0x200⏎	S_ISUID                           = 0x800⏎	S_ISVTX[[                        ]]   = 0x` → '   ', help -0.0
- `_CRITICAL                     = 0x20⏎	NOTE_DELETE                       = 0x1⏎	NOTE_EXEC[[                        ]] = 0x2` → ' =', help -0.018

**Facet 3, tables: aligned columns** (40 examples, 6 from v1): harmless, median help -0.0 (p = 0.5185), later predictions -0.0 (harmless); reliability 0.884, generic share 0.351

- probability: median KL 0.000119, moved 0.001736, sharpen cosine -0.019; raises '     ', '.', '    ', '   ', '       ', '3', ',', '  '; lowers '      ', ' it', '                        ', '-', '        ', '/', '2', ' \n\n'
- promoted: 'iper', '�', 'ώ', 'imeter', 'Frames', 'ring', 'aho', 'etra'
- specific promoted: 'imeter', '�', 'nem', 'himself', '\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0', 'ité', 'stop', 'eLife'
- specific demoted: '.', '!', '!!', 'han', 'trl', ' Bl', 'velt', '~~'
- ` Å       *T* = 120 K⏎  *V* = 1844.87 (14) Å^3[[^]]   Polygon shape` → '   ', help -0.0
- `A)                                                **Total energy stored**   **10,740 kWh**⏎  I~mpp~[[                        ]]     8.10` → '     ', help 0.125
- `.1315 (0.1917)       --0.1447 (0.1925)⏎  GT[[                        ]]    --0.` → '    ', help -0.0

- pair 0-1: no use found; score 0.993, specific cosine 0.842, raw cosine 0.897
- pair 0-2: no use found; score -0.131, specific cosine -0.11, raw cosine 0.23
- pair 0-3: no use found; score -0.033, specific cosine -0.029, raw cosine 0.333
- pair 1-2: no use found; score -0.149, specific cosine -0.121, raw cosine 0.222
- pair 1-3: no use found; score -0.044, specific cosine -0.038, raw cosine 0.326
- pair 2-3: no use found; score 0.539, specific cosine 0.455, raw cosine 0.619

## Neuron 4052: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.718; patched into random text at that level, its median help is -0.0, KL 0.00052, sharpen cosine -0.0.

**Facet 0, facet 0** (40 examples, 7 from v1): harmless, median help 0.0003 (p = 0.6273), later predictions -0.0002 (harmless); reliability 0.734, generic share 0.423

- probability: median KL 0.000693, moved 0.013644, sharpen cosine 0.035; raises ' this', ' a', ' ex', ' under', ' Relief', ' on', ' to', ' relief'; lowers ' that', ' in', ' we', ' mistrial', ' the', ' evidence', ' it', ' his'
- promoted: 'Power', 'arie', 'unless', 'oir', ' essential', 'arial', 'power', ' precedence'
- specific promoted: 'Unless', 'Pb', 'Append', 'emer', 'FPar', 'Any', 'Conflict', ' quoting'
- specific demoted: 'agues', 'iates', 'rich', 'yll', 'egan', 'documentclass', ' Plans', 'aign'
- ` by police who wanted to prevent him attending a May Day⏎rally. They had neither a court order nor warrant to[[ place]] him under house⏎` → ' him', help -0.003
- ` of the trial court should be AFFIRMED. The Court⏎orders the judgment of the trial court AFFIRMED.⏎⏎      We further[[ order]] this decision certified below` → ' this', help 0.087
- ` has attempted to demonstrate diversity by alleging in its Complaint that Plaintiff is incorporated in Nebraska and that "Defendant is an Indian Tribe[[ with]] its principal place of` → ' its', help 0.002

**Facet 1, facet 1** (11 examples, 7 from v1): harmless, median help -0.0154 (p = 0.3203), later predictions -0.0062 (harmless); reliability 0.789, generic share 0.375

- probability: median KL 0.00145, moved 0.02199, sharpen cosine 0.021; raises 'id', 'input', 'a', ' style', 'class', 'd', 'ul', '['; lowers 'td', 'div', 'data', 'name', 'style', 'title', 'active', ' object'
- promoted: 'wy', 'onas', 'riter', 'acs', 'on', 'ippets', ' Harper', 'uance'
- specific promoted: 'uance', 'wy', 'gments', ' Nolan', 'clic', 'lys', 'ippets', '-'
- specific demoted: '*^', 'ATIONAL', 'ational', ' somet', 'endant', 'OPER', '}?', 'ocrates'
- `)') , index );⏎            var id = 'id_' + name;⏎            $(this).attr({[[']]name': name,` → 'name', help -0.028
- `    _renderItem: function(ul, item) {⏎        return $("<li>")⏎            .attr[[("]]data-value",` → 'data', help -0.061
- `                new_id = new_id.replace('zero', some_id);⏎                $(this).attr({[[']]id' : new` → 'id', help 0.001

**Facet 2, facet 2** (13 examples, 5 from v1): harmless, median help 0.0104 (p = 0.4143), later predictions -0.0008 (harmless); reliability 0.749, generic share 0.438

- probability: median KL 0.000867, moved 0.016474, sharpen cosine 0.018; raises 'background', 'left', 'right', 'display', ' class', 'next', 'width', ' data'; lowers 'top', 'font', 'margin', ' href', 'float', 'color', 'step', 'text'
- promoted: 'unity', 'iegel', 'onitrile', 'rics', 'compact', 'ks', 'ogi', 'acy'
- specific promoted: 'iegel', 'oyo', 'pts', 'uance', 'opan', 'exposed', 'compact', ' infinity'
- specific demoted: 'enron', 'urer', '.";', '|\\', '”.', ']{};', ' pub', 'Month'
- `size:16px;  /* 16*1.5= 24 */⏎}⏎⏎.btn {⏎[[  ]]font-size:` → 'font', help -0.063
- `        $(truck[0]).css('left', css_left);⏎        $(truck[1]).css[[(']]right', css_` → 'right', help 0.083
- ` alignment of text within div(Possibility without using flex property)⏎⏎Thanks in advance :)⏎⏎div{⏎[[  ]]width:100px` → 'width', help 0.026

- pair 0-1: no use found; score -0.071, specific cosine -0.054, raw cosine 0.366
- pair 0-2: no use found; score -0.209, specific cosine -0.155, raw cosine 0.343
- pair 1-2: no use found; score 0.732, specific cosine 0.563, raw cosine 0.739

## Neuron 1710: v1 packing-like, v2 packing-like, v3 no use found (facets 0-1)

Facet-typical activation 1.847; patched into random text at that level, its median help is -0.0002, KL 0.000743, sharpen cosine 0.003.

**Facet 0, travel: prices** (40 examples, 25 from v1): harmless, median help -0.0035 (p = 0.5185), later predictions 0.0003 (harmless); reliability 0.974, generic share 0.144

- probability: median KL 0.000543, moved 0.012175, sharpen cosine 0.118; raises '.', ')', ',', '),', ' return', ');', ').', ' &'; lowers '/', '-', '50', '00', ';', '40', '–', '30'
- promoted: ' simultaneous', ' combined', ' Sau', 'ults', ' pools', 'ourses', ' single', ' TOD'
- specific promoted: ' Sau', 'ults', ' Ts', ' Trent', ' Tai', ' Ser', ' Fac', ' Tu'
- specific demoted: '~\\', '...]', 'orc', '?!', 'empre', 'stats', '8', '83'
- ` served sizzling on a hot lava rock. The three-course lunch and sharing boards (£15.30 to £[[18]].50 for four` → '.', help 0.019
- `; www.glencoehostel.co.uk; dm £13-16.50, bunkhouse £[[12]].50-14` → '.', help 0.008
- `.lindenguesthouse.co.uk; 22 Linden Ave; d £65-80, f £[[90]]-150;  ` → '-', help -0.023

**Facet 1, fiction: 'nodded/shrugged'** (40 examples, 6 from v1): harmless, median help -0.0004 (p = 0.4762), later predictions 0.0006 (harmless); reliability 0.922, generic share 0.417

- probability: median KL 0.000562, moved 0.011513, sharpen cosine 0.005; raises ' and', ' at', ',', ' me', ' that', ' off', "'t", '."'; lowers '.', ' my', ' his', ' as', ' no', ' in', ' her', ' a'
- promoted: ' discussion', ' debate', 'pts', 'urg', ' DIRECT', 'demand', 'graphs', ' torped'
- specific promoted: ' Orders', ' accusation', ' charges', ' orders', ' discussion', ' torped', 'graphs', ' vengeance'
- specific demoted: 'conc', 'concat', 'ipheral', '�', 'seud', 'itos', 'Add', 'figure'
- ` there was something big in my throat blocking it. He relaxed his hands and let me go. "Understand?" I[[ nodded]] mutely. "` → ' mut', help 0.12
- ` even though I didn't feel like it, and forced myself to eat.⏎⏎'Then what?'⏎⏎I[[ shrugged]]. I couldn't` → '.', help 0.016
- ` into the kitchen. "You're doing fine, just take a deep breath and let's go."⏎⏎Jackie[[ nodded]] and fixed a smile` → ' and', help 0.006

- pair 0-1: no use found; score 0.064, specific cosine 0.061, raw cosine 0.288
