# v8 results per model

Flag rates (unrelated jobs) by reader and unit type at the middle layer, then, per model, up to 2 units of each kind worth reading: 50/50 pools an agent called one job, and real neurons the agents disagree on. For each, every reader's label, the agents' stated patterns and their tool calls (for test texts, the unit's strongest activation on each, on the 0-10 scale). Unit ids use v6's sample indices; the MLP neuron ids are given beside them.

## pythia-70m

| Reader | real | pooled 50/50 | related |
|---|---|---|---|
| minilm | 20% | 8% | 5% |
| network | 48% | 78% | 55% |
| haiku | 40% | 50% | 25% |
| haiku agent | 22% | 35% | 10% |
| sonnet agent | 57% | 72% | 40% |

### 50/50 pools the Haiku agent calls one job

**70m:pooled:50/50:424-1001**, MLP neuron(s) 543, 1306: haiku A, haiku agent A, sonnet agent C, network True, MiniLM False

- haiku agent (A, low confidence): formulaic verb/preposition slot in acknowledgments and set phrases ("thank", "grateful to", "indebted to", "Due to") where the next token is highly predictable
  - examples 51-70
  - activation on "The authors would like to thank the reviewers for their comm" (8.8); "We are grateful to the nurses who helped with data collectio" (6.6); "Due to heavy rain, the match was postponed." (9.1); "The company announced that it would expand into new markets " (3.0); "Je ne peux laisser entrer personne : les sept nains m'ont dé" (2.6)
- sonnet agent (C, medium confidence): acknowledgment phrases: 'thank', 'grateful/indebted to' / clause-opening function words: 'Due', 'Because', 'whose', 'using', 'to' / French l' elision and genitive 's apostrophes / code keywords like #include / use / word-fragment prefixes in Hungarian/Spanish, and math-formatting tokens
  - activation on "The authors thank Dr. Smith. We are grateful to John." (8.6); "Il l'a vu et je l'ai dit." (5.1); "Due to the rain, we left." (9.1); "#include "foo.h"
use Foo\Bar;" (6.3)
  - examples 100-119

**70m:pooled:50/50:420-746**, MLP neuron(s) 538, 970: haiku A, haiku agent A, sonnet agent B, network True, MiniLM False

- haiku agent (A, medium confidence): the word completing or extending a fixed multi-word collocation, such as 'represented/reported/described as', 'motion for', 'thank you very much', 'with respect to', 'according to', 'something like this' / formulaic phrase-internal tokens across legal, scientific, dialogue and tutorial text, where the next word is highly predictable
  - examples 51-70
  - activation on "Data are represented as mean ± SD." (6.0); "Thank you very much." (6.2); "The defendant filed a motion for summary judgment." (4.6); "I respectfully dissent from the opinion." (4.8); "She ate an apple and then went to sleep." (2.6)
- sonnet agent (B, low confidence): Function words after a verb or noun that set up a formulaic or predictable continuation: 'motion for', 'represented/reported as', 'recommend that', 'according to', 'various', 'following' / Formulaic dialogue and legal phrases: 'Thank you very much', 'I respectfully dissent', 'I will/would'
  - activation on "Data are represented as mean" (6.0); "The motion for summary judgment" (1.0); "Thank you very much." (6.2); "I respectfully dissent" (4.8); "He went to the store for milk" (2.6)
  - examples 100-114
  - activation on "She recommended that we go" (6.3); "It looks something like this" (2.5); "according to the report" (0.8); "under various conditions" (3.5); "I will allow no delay" (3.5)
  - examples 300-314

### 50/50 pools the Sonnet agent calls one job

**70m:pooled:50/50:714-1355**, MLP neuron(s) 931, 1775: haiku A, haiku agent A, sonnet agent A, network False, MiniLM False

- haiku agent (A, high confidence): past participle (or verb) immediately following a passive auxiliary 'is/was/are/were', especially evaluative or procedural participles like 'deemed', 'considered', 'introduced', 'applied', 'used', 'provided' in academic/technical passive constructions
  - activation on "The results were deemed unreliable by the reviewers." (6.5); "She deemed the results unreliable." (6.2); "The bridge was built in 1920 and is still used today." (0.1); "A new method is introduced in this section." (4.4); "The dog ran quickly across the yard." (-0.1)
  - examples 50-64
- sonnet agent (A, medium confidence): Passive or copular verb right after a subject noun plus auxiliary (is/are/was/been), where the verb is a participle such as considered, deemed, provided, introduced, used or applied, in formal or technical prose
  - activation on "A voltage is applied to the line." (3.1); "He applied the voltage to the line." (-0.1); "The method is considered safe." (3.8); "They considered it safe." (4.4); "The dog was happy and is running." (-0.1)
  - examples 100-114

**70m:pooled:50/50:1375-644**, MLP neuron(s) 1805, 840: haiku A, haiku agent A, sonnet agent A, network False, MiniLM False

- haiku agent (A, medium confidence): Connector or operator token inside a parallel or templated structure, where the next token is predictable from the repeated pattern: number ranges like "5 to 10", "X in the group, Y in", "one for ..., one for ...", "30 for 30", and arithmetic or code operators like "-", "*", "+" in repeated expressions.
  - activation on "The study enrolled 120 patients: 60 in the treatment group, " (4.0); "Solve for x: 3*x - 7 = 11 - 2*x. x = 3" (6.5); "He ran a 5 to 10 mile route, then a 10 to 15 mile route, the" (7.9); "The weather today is sunny and warm, with a light breeze fro" (1.1); "def f(a, b):
    return a[n - 1] * b[n + 1] - c" (8.0)
  - examples 20-29
- sonnet agent (A, medium confidence): connector or separator token (comma, 'to', 'of', operator, repeated 'one') inside a list, parallel structure, or fixed 'X to Y' phrase where the next item is expected
  - activation on "The variables were: age, BMI, height, weight, and sex." (5.7); "ESPN aired the doc as part of its 30 for 30 series." (3.0); "Solve 4*r = 7*r - 132 for r." (8.0); "The cat sat on the mat and looked at me." (0.5)
  - examples 100-109
  - activation on "He bought one apple, one pear, one plum." (4.2); "Go from 10 to 20 miles, Coast to Coast, door to door." (5.2); "x = a[n - 1] * b[n + 1]" (6.1); "She walked home slowly." (1.2)
  - examples 400-409

### Real neurons the agents disagree on

**70m:real::56**, MLP neuron(s) 78: haiku C, haiku agent A, sonnet agent C, network False, MiniLM False

- haiku agent (A, medium confidence): Highly formulaic or templated text where the next token is nearly fixed: sign-offs and signature breaks ("Kind regards,⏎", "Thanks,⏎"), "look forward to" closers, citation boilerplate ("Kluwer Academic Publishers,"), web page boilerplate ("Like this:⏎", "Share this post⏎"), and auto-generated math problems (derivative of a polynomial, "prob of picking"), where the unit fires on the predictable token
  - examples 51-70
  - activation on "Let q(m) be the first derivative of -m**3/3 - 6*m**2 - 27*m " (5.8); "Thanks,

Marc

Kind regards,

Talib" (8.3); "We look forward to seeing you soon." (5.7); "Kluwer Academic Publishers, Boston, 2000." (2.6); "What is prob of picking 1 u and 1 p? Calculate prob of picki" (6.1)
  - activation on "The capital of France is Paris, and the capital of Japan is " (1.9); "I went to the store yesterday and bought a loaf of bread, so" (1.9); "Let f(x) be the second derivative of x**4/12 - x**3/2. Find " (4.0); "Best regards,

Sarah Johnson
Senior Analyst" (5.2); "Thank you for your order. We look forward to serving you aga" (6.2)
- sonnet agent (C, medium confidence): newline after sign-offs or web boilerplate headers (Thanks,⏎; Posted by X⏎; Share this post⏎) / exponent/multiplication operators in polynomial and derivative math problems (**, *) / comma after publisher or city in bibliography citations / "forward" or "to" in "look forward to", and "you" in "Thank you"
  - activation on "Let q(y) be the second derivative of -y**4/3 + y**3/3 - 5" (4.9); "Thanks,

Marc" (4.9); "Kluwer Academic Publishers, Boston, 2000." (2.6); "We look forward to seeing you" (5.7); "The cat sat on the mat." (0.2)
  - examples 100-114
  - activation on "Sincerely,
John" (3.2); "Posted by Bob

Hello" (5.0); "x = 3*y**2 + 4*z**3" (-0.1); "Springer, New York, 1999." (2.8); "Share this post

Link" (3.6)
  - examples 300-314

**70m:real::60**, MLP neuron(s) 83: haiku C, haiku agent A, sonnet agent C, network False, MiniLM False

- haiku agent (A, low confidence): Line-ending token (usually a double newline) at the end of a complete statement inside a block of source code, where the next line continues the same code body
  - examples 51-70
  - activation on "int main() {
    int x = 0;

    return x;
}" (5.4); "The patients were enrolled in the study. A retrospective coh" (3.6); "Give -3*b(s) + 5*y(s). Let a(j) = 2*j. Give 5*a(j) + 24*b(j)" (6.0); "    var options = Object.extend(defaults, arguments[1] || { " (6.9); "The cat sat on the mat and the dog barked loudly at the mail" (-0.1)
- sonnet agent (C, medium confidence): blank line (double newline) after a statement or closing brace in indented source code / verb like 'was' after 'study' in medical abstracts / '+' operator in math problem expressions
  - activation on "def f():
    x = 1

    return x" (5.5); "A prospective study of 570 patients was undertaken." (3.7); "Give -3*b(s) + 5*y(u)" (1.6); "Hello there.

New paragraph here." (0.8)
  - examples 100-114
  - examples 1000-1014
  - activation on "int main() {
    int a = 0;

    return a;
}" (5.4); "A case-control study was conducted in which 45 patients" (4.2); "Let a(x) = 3*x. Give 5*a(j) + 24*b(k)" (3.9)

## pythia-410m

| Reader | real | pooled 50/50 | related |
|---|---|---|---|
| minilm | 5% | 15% | 15% |
| network | 52% | 85% | 50% |
| haiku | 48% | 62% | 45% |
| haiku agent | 20% | 30% | 20% |
| sonnet agent | 52% | 85% | 40% |

### 50/50 pools the Haiku agent calls one job

**410m:pooled:50/50:36-140**, MLP neuron(s) 63, 264: haiku B, haiku agent A, sonnet agent C, network True, MiniLM True

- haiku agent (A, medium confidence): Token inside a cross-reference or label sequence (figure/panel/table/section/statute citations such as 'Fig 2A and B', '(A--G)', '5 U.S.C', 'Section 3.2.4'), where the next token continues the label / Operator or connector token inside math exponent/range notation (e.g. 'x**(-2/3)', '**--**') that precedes another symbol or number
  - activation on "As shown in Figure 2A and B, the mutant cells displayed redu" (7.2); "The court relied on 5 U.S.C. § 706 in reviewing the agency's" (6.1); "The government's ability to intercept messages was widely de" (1.0); "She bought apples and oranges at the market on Tuesday." (0.7); "Results are shown in panels (A--G) of the figure above." (5.5)
  - examples 51-70
  - activation on "Simplify (x**5)*(x**(-2/3)) and then differentiate the resul" (6.5); "The function is f(x) = 3*x**2 + 2*x**3 - x**4." (1.7); "We went to the store and bought milk, eggs, and bread before" (-0.1); "See Table 2 and Figure 4, and the supplementary material in " (7.6)
- sonnet agent (C, medium confidence): possessive nouns missing their apostrophe (presidents, Dads, Marks, governments) / 'and' or '--' joining panel labels in figure references (Fig. 2A and B, A--E) / 'and' after a subject noun phrase (the city and the site) / mid-word fragments and code/citation tokens (KcsA, U.S.C., Configuration)
  - activation on "He took the presidents decision to my Dads house." (7.6); "as shown in (Fig. 2A and B) and (A--E) panels." (7.8); "The city and the site were large." (8.0)

**410m:pooled:50/50:1920-1959**, MLP neuron(s) 3846, 3919: haiku B, haiku agent A, sonnet agent C, network True, MiniLM False

- haiku agent (A, low confidence): the conjunction 'and'/'or' inside a list of abstract attributes, just before the final item (e.g. 'accuracy and', 'WITHOUT WARRANTIES OR') / a token inside a fixed or formulaic phrase whose continuation is highly predictable ('Originally Posted by', 'For the full copyright and', 'Matter of') / a newline or punctuation closing a sign-off or forum post, just before a name or next block ('Cheers,⏎', 'God bless,⏎')
  - examples 51-70
  - activation on "The company designed and built the bridge, and the engineers" (4.3); "Originally Posted by Mbeck

I agree with you on that." (5.9); "Thanks for reading!

Posted by veritas

" (3.1); "Our product offers speed, accuracy and reliability for every" (7.8); "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express" (6.9)
  - activation on "She bought apples, oranges, and" (1.0); "The results were significant and" (3.0); "Please sign in or register to continue reading the article." (2.9); "In the case of Smith v. Jones, the court held that the statu" (2.4); "He ran quickly, jumped high, and" (1.3)
- sonnet agent (C, medium confidence): Conjunctions or commas in lists of parallel nouns or verbs, such as 'design and construction', 'copy, modify, merge', and 'WARRANTIES OR CONDITIONS' / Blank-line newline tokens between forum comments, or after a sign-off such as 'Cheers,⏎' or 'God bless,⏎' / 'by' in 'Originally Posted by', 'of' in 'out of' or 'Matter of', and closing quotes after legal citations
  - activation on "We improve the design and construction of tools.

Hello ther" (7.5); "to use, copy, modify, merge, publish, distribute, sublicense" (10.2); "Originally Posted by Bob" (5.9); "I like cats and dogs." (0.5)
  - examples 100-114
  - examples 1000-1014
  - activation on "The system supports sending, receiving, and storing data." (8.1); "Thanks for reading.

Best,
John" (5.8); "Matter of Smith, and see also" (1.0)

### 50/50 pools the Sonnet agent calls one job

**410m:pooled:50/50:1155-1264**, MLP neuron(s) 2294, 2519: haiku A, haiku agent A, sonnet agent A, network False, MiniLM False

- haiku agent (A, medium confidence): Hyphen or dash that joins a compound modifier or name before its second part (e.g. 'North Carolina-based', 'T-shirt', 'Al-Sanadid', '16-bit', '200-fold'), where the next token completes the compound / Mid-token splits inside identifiers, numbers or hyphenated terms where the next piece is still part of the same word (e.g. 'A27[[80]]', 'm[[12]]', '[[ 412]]55')
  - activation on "The company is North Carolina-based and makes T-shirts." (6.5); "The 16-bit converter uses a 252-Cf source and a 2-fold incre" (6.6); "We measured the temperature-dependent coefficient in the sam" (3.1); "The cat sat on the mat and looked out the window at the rain" (1.6); "Presidente
 Señora Lopez, gracias." (2.9)
  - examples 30-49
  - activation on "Stats are exclusive. Top horse: Races 286, Win Percent 12." (1.0); "He was a well-known, high-quality, long-term investor." (1.1); "The data were collected in 1995-2000 across three sites." (2.5); "She bought a new car and drove it home to the coast." (0.1)
  - examples 1-5
- sonnet agent (A, medium confidence): hyphen or separator inside a short-prefix compound name (T-Mobile, Al-Ahram, D-Wave) / repeated/enumerated structure tokens in lists (STAT5 and STAT6, Race N, line N) / mid-identifier or number pieces where the continuation is predictable
  - activation on "T-Mobile and D-Wave and Al-Sanadid" (6.2); "a well-known fact about state-of-the-art work" (2.5); "STAT4, STAT5 and STAT6" (4.1); "the 16-bit value" (2.2)
  - examples 100-114
  - activation on "Race 1, Race 2, Race 3 and Race 4" (4.9); "type I, type II, and type III" (2.7); "The cat sat on the mat." (0.3); "line 3, line 4 and line 5" (4.3)
  - activation on "Presidente
Señora" (3.7); "A2780, A2780cis and A2780-CP20" (4.4); "0.02, 0.04, 0.06" (3.3)

### Real neurons the agents disagree on

**410m:real::18**, MLP neuron(s) 31: haiku A, haiku agent A, sonnet agent C, network True, MiniLM False

- haiku agent (A, low confidence): tokens inside a highly predictable continuation, such as a word split across tokens, a fixed collocation, or a formulaic phrase (e.g., 'the last time', 'delivered to your inbox', 'At that moment')
  - examples 51-70
  - activation on "It was the last time I saw him alive." (6.1); "The team reached the last round of the tournament." (3.2); "Ich bin mir nicht sicher, ob das stimmt." (5.5); "def foo(x):
    return x + 1" (-0.4); "The results were significantly better than the baseline." (1.8)
  - activation on "The unbelievably long-winded explanation was incomprehensibl" (1.6); "Please send the final draft by the last day of the month." (3.6); "He was the last person to leave the building." (4.1); "Ich bin mir nicht sicher, ob das stimmt. Wir werden sehen, w" (5.5); "Fjallet var hohet og den ble kald om vinteren." (2.9)
  - examples 71-85
  - activation on "Get our best stories delivered to your inbox every morning." (5.3); "At that moment, the door opened and the guard stepped in." (3.9); "The quick brown fox jumps over the lazy dog." (1.5); "Thank you for your order. Click this link to confirm your em" (1.9)
- sonnet agent (C, medium confidence): "last" after "the" (the last time/row/line) / mid-word suffix or fragment tokens, such as -ating, -ing, -ered and Danish " s" in sikker/sige / "the" in "all the better" or "all the more" / code and format fragments such as "__" in __be16 and "#!" shebang / other function words in the middle of a phrase, such as "your", "these" and "this"
  - activation on "It was the last time I saw the last row, and all the better " (6.1); "Plastic scintillating plates and a segregating SNP in the pr" (3.2); "Ich bin nicht sicher, det er lykkedes. Jeg vil gerne sige" (7.6); "#!/bin/sh
__be16 local_port;" (2.2)
  - examples 51-70

**410m:real::135**, MLP neuron(s) 255: haiku B, haiku agent A, sonnet agent C, network True, MiniLM False

- haiku agent (A, medium confidence): Function word (mainly 'by', 'in', 'to', 'than', 'of') ending a phrase in academic or technical prose, right before a predictable continuation such as a method, a citation, or a section/theorem reference
  - examples 51-70
  - activation on "In Section " (2.9); "The case was proven by Pukhlikov in " (2.8); "We start by" (3.8); "The results were better than" (-0.3); "As shown in " (1.7)
  - activation on "The sum is more than" (0.0); "He decided to" (0.5); "We solve this problem by" (2.1); "The proof follows from Lemma" (0.2); "Smith et al. showed this in" (1.9)
- sonnet agent (C, medium confidence): prepositions/particles following a verb or quantifier (start by, more than, equivalent to, amounts to) / space token right before a citation or reference marker ([@..., \[sec:...) / mid-word fragments in rare names and chemical terms (pyrrolo, Samhain, Graim)
  - activation on "We start by removing the items. This is more than 10." (3.8); "As shown by [@Spirtes2000] and in Section \ref{a}." (3.2); "N-methyl-2-pyrrolidone and Samhain" (6.3); "He went to the store and bought milk." (1.9)
  - examples 100-114

## pythia-2.8b

| Reader | real | pooled 50/50 | related |
|---|---|---|---|
| minilm | 2% | 12% | 0% |
| network | 42% | 50% | 35% |
| haiku | 45% | 78% | 25% |
| haiku agent | 30% | 40% | 5% |
| sonnet agent | 57% | 85% | 30% |

### 50/50 pools the Haiku agent calls one job

**2.8b:pooled:50/50:1085-627**, MLP neuron(s) 5432, 3062: haiku C, haiku agent A, sonnet agent C, network False, MiniLM False

- haiku agent (A, low confidence): token completing a fixed or highly predictable phrase or verb-complement (e.g., 'bear with me', 'trust me', 'leave them alone', 'Editorial assistance', 'the other way')
  - examples 51-70
  - activation on "Just trust me on this." (6.6); "Please bear with me while I fix it." (8.8); "$POST = array_intersect" (1.0); "The study was supported by funding from Pfizer. Editorial as" (6.8); "Leave them alone." (4.6)
- sonnet agent (C, low confidence): Object or completion after an imperative or relational verb phrase in dialogue or prose, e.g. 'trust me', 'bear with me', 'leave them alone', 'stay with me', 'forget X for a sec' / Assignment '=' after a variable name in code (PHP/JS) / Scattered tokens in legal or technical prose, e.g. 'state law', 'only', 'property', and word-fragment tokens
  - activation on "Just trust me on this." (6.6); "csv = values.join();" (2.6); "Leave them alone." (4.6); "The cat sat on the mat." (1.2)
  - examples 100-114
  - activation on "var msg = $('#message').val();" (-0.0); "Bear with me, please." (7.4); "state law merely because" (-0.1); "Forget polar bears for a sec" (8.1)
  - examples 300-314

**2.8b:pooled:50/50:1310-1307**, MLP neuron(s) 6607, 6580: haiku A, haiku agent A, sonnet agent C, network True, MiniLM False

- haiku agent (A, medium confidence): Mid-clause word (usually a verb, or a preposition/determiner) that opens a slot for an upcoming object or complement, e.g. 'which [owns] the largest...', 'that [sell] software', 'trusted me [with] tasks', 'told [reporters]', 'said' after a quote
  - examples 51-70
  - activation on "The company, which owns the largest chain of stores in the r" (6.4); "The company, which owns the largest chain of stores in the r" (6.4); "She trusted me with tasks like that, and she told reporters " (6.4); "The number 7 is prime. 2 + 2 = 4. The cat sat on the mat." (1.7); "He walked the ball up the court, and the Russians have dirt " (4.0)
  - activation on "The hotel that offers free breakfast is near the station." (4.8); "The weather today is sunny and warm, and the sky is blue." (1.1); "Many companies that sell software to schools have grown fast" (5.1); "She is a member of the committee and a friend of many people" (0.7); "The river flows north. Birds sing in the morning. Rain fell " (0.9)
  - examples 71-82
- sonnet agent (C, low confidence): transitive verbs right after a subject or relative pronoun (that stores, which owns, he told, Payton said) / numbers or digits inside chemical names, math expressions and legal paragraph or footnote citations / prepositions, quantifiers and adverbs mid-phrase (much of, own, even, more)
  - examples 50-69
  - activation on "a network node that stores a copy of the data" (7.5); "Payton said, via the New York Times" (0.8); "-1788*f**2 - 24*f + 24" (3.4); "The family, which owns the team, said" (5.4)

### 50/50 pools the Sonnet agent calls one job

**2.8b:pooled:50/50:246-374**, MLP neuron(s) 1102, 1771: haiku C, haiku agent A, sonnet agent A, network False, MiniLM False

- haiku agent (A, medium confidence): Token where the next word is predictably the second half of a coordinated or paired construction (e.g., "input and [output]", "reactive oxygen species and [antioxidant]", "input/[output]") / Token before a comparison or quantity complement (e.g., "significantly [more]", "more than [X]", "spent [over]")
  - examples 51-70
  - activation on "The device includes an input and" (7.3); "The device includes an input/output" (6.8); "Oxidative stress is an imbalance between reactive oxygen spe" (8.1); "The cat sat on the mat and the dog" (0.8); "Gerbils spent significantly more time sniffing than" (6.3)
- sonnet agent (A, medium confidence): connector or comparative token (and, /, than, to, of) in a paired or contrasting structure, where the text is about to give a counterpart of an earlier term (input/output, produced and consumed, more than we export) / repeated or parallel structure continuation, including code and table repetition
  - activation on "The system has an input and output stage; we import more tha" (10.7); "The cat sat on the mat and looked at the dog." (0.8); "Supply and Demand, input/output, production and consumption" (7.6)
  - examples 60-69
  - activation on "Gerbils spent significantly more time sniffing than resting." (6.3); "energy flows from the source to the scale where it is dissip" (7.2); "a single-phase alternating current input and a three-phase a" (6.9)
  - activation on "Model 10: seconds 9.99" (0.3); "uint32_t src, uint32_t dst" (6.9); "He went to the store to buy milk and eggs." (0.3)

### Real neurons the agents disagree on

**2.8b:real::202**, MLP neuron(s) 930: haiku B, haiku agent A, sonnet agent C, network True, MiniLM False

- haiku agent (A, medium confidence): Mid-word subword fragment of a longer word split across tokens (e.g., 'ps' in dyspepsia, 'eps' in catalepsy, 'ots' in Cotswolds/Botswana, 'aps' in synapsis, 'ars' in arsB)
  - examples 50-69
  - activation on "The Cotswolds offers endless walks." (6.1); "Botswana is a country in Africa." (4.6); "The catalepsy reversal was observed." (9.2); "He pulled out of the driveway." (3.9); "She said the phrase was eps-matches in the output." (6.6)
  - activation on "The doctor diagnosed dyspepsia and prescribed rest." (7.9); "The cat sat on the mat and looked out the window." (3.8); "Seqs with Stellar eps-matches and mutations listed." (6.7); "Synapsis of chromosomes occurs in pachytene." (7.6); "The quick brown fox jumps over the lazy dog." (0.4)
- sonnet agent (C, medium confidence): mid-word subword fragments such as eps, ots, ps, ars, ens, am (Botswana, Cotswolds, catalepsy, dyspepsia, Stellar eps-matches) / the particle 'out' after verbs (pull out, radiating out, get it out there) / LaTeX '}{\' after \frac
  - activation on "seq1	Stellar	eps-matches	970" (6.6); "The Cotswolds and Botswana" (6.1); "He pulled out of it and headed out" (4.6); "dyspepsia and catalepsy" (9.8)
  - examples 50-64

**2.8b:real::221**, MLP neuron(s) 1016: haiku C, haiku agent A, sonnet agent C, network True, MiniLM False

- haiku agent (A, low confidence): token in technical, scientific, or code text (chemical names, math expressions, citations, SSH/compiler messages) where the next token is tightly constrained by notation or syntax
  - examples 51-70
  - activation on "We found that the treatment reduced tumor size. We then meas" (0.9); "The number 5 appears here: 5*x + 5*y = 10." (2.5); "He said that the results were good, and the team was happy." (0.5); "Galβ3/4(Fucα4/3)Glc structures are common in glycans." (6.6); "known hosts. Warning: the ECDSA key for host was added to th" (7.1)
- sonnet agent (C, low confidence): mid-word or continuation tokens in chemical/biological names and identifiers, often after digits like 5- / function words and connectors in formulaic or boilerplate text (of, to, We, by, otherwise) / separators and punctuation in code, numbers, or formulas (/, @, _, commas)
  - activation on "We found that the results were good. We report here that" (0.6); "Galβ3/4(Fucα4/3)Glc and 5-hydroxytryptamine and 5% of the ca" (7.7); "No more authentication methods to try. added the key to the " (5.9)
  - examples 100-119
