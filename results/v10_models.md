# v10 results per model

Flag rates (unrelated mixed) by reader and unit type at the middle layer, the language-model readers' labels on real neurons, then, per model, up to 2 50/50 pools where Opus and Haiku disagree in each direction and 2 real neurons where they disagree, with each reader's stated patterns and four of the unit's 50 snippets (its 1st, 2nd, 26th and 27th strongest; the readers saw all 50 in random order).

## pythia-70m

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 17% | 16% | 20% | 10% | 6% |
| network | 42% | 82% | 74% | 62% | 44% |
| haiku | 38% | 56% | 52% | 42% | 22% |
| sonnet | 33% | 64% | 56% | 44% | 28% |
| opus | 31% | 60% | 62% | 42% | 16% |

haiku labels on real neurons: A 53, B 9, C 38

sonnet labels on real neurons: A 11, B 56, C 33

opus labels on real neurons: A 38, B 31, C 31

### 50/50 pools Opus flags and Haiku calls one job

**70m:pooled:50/50:424-1001**, MLP neuron(s) 543, 1306 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["Token at a point where the next word is highly predictable from a fixed collocation or elision, such as 'thank' after 'the authors would like to', 'to' after 'grateful'/'indebted', 'Due' opening 'Due to', and French or Italian clitic and elision splits like l[[']] or ha[[i]]"]}`
- sonnet: `{"label": "C", "patterns": ["acknowledgment verbs/prepositions after 'thank', 'grateful', 'indebted' (thank, to)", "apostrophes/elisions and possessive 's in French and other languages (l', 's)", "word-start fragments in Hungarian/Spanish/Russian/Italian (Az, al, ha, при)", "sentence-initial 'Due' before 'to'", "code tokens like include, use, quote openings, and newline in math"]}`
- opus: `{"label": "C", "patterns": ["'thank' / 'grateful to' / 'indebted to' in paper acknowledgment sections", "French elided article l' before a vowel-initial word", "sentence-initial connectives such as 'Due', 'Because' and 'In spite'", "possessive 's after a proper noun", "code tokens such as #include, use and opening parentheses", "word-fragment pieces in Hungarian, Spanish and Russian ('Az', 'al', 'при')", "newline after a math question, before the answer"]}`
- ` could be completely inert.⏎⏎Finally, in the spring of 1900, at a meeting in Berlin, Ramsay suggested[[ to]] Mendeleev`
- `.⏎⏎(XLSX)⏎⏎###### ⏎⏎Click here for additional data file.⏎⏎The authors[[ thank]] Katsunori`
- `d**3 + 5937*b*d + 48989*b + 19*d**4 wrt d.[[⏎]]3*b*`
- `udnak a problémáról, és dolgoznak azon, hogy kijavítsák.[[ Az]]t javasol`

**70m:pooled:50/50:975-1500**, MLP neuron(s) 1277, 1965 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["token right before a highly predictable, formulaic continuation (stock phrases, idioms, or fixed code/number syntax)"]}`
- sonnet: `{"label": "C", "patterns": ["open parenthesis after 'for' in loop headers in code", "patent-style phrasing: 'provided', 'directed to', 'invention to/provides', 'present invention'", "word after 'a/different/two' like 'closer', 'types', 'while'", "'?' ending number sequence questions", "whitespace padding tokens in legal text"]}`
- opus: `{"label": "C", "patterns": ["opening '(' after 'for' in code loops", "formulaic patent phrasing ('there is provided', 'directed to', 'present invention', 'object of the invention to')", "'a closer/clearer' before look/view/eye", "'types' after 'different/two/various/other'", "'?' ending number-sequence questions", "whitespace runs in legal court-opinion text", "'for a while'"]}`
- ` the period of time of an interruption in a program comprising commercials will be reduced.⏎It is an object of the invention[[ to]] provide a device having`
- `, especially at points of substrate imperfection. The tendency of a coating to shrink away from a substrate surface results in various[[ types]] of coating defects,`
- `-dependent IVD changes. In addition to being a reliable model of IVD aging, this spontaneous model may provide a[[ closer]] representation of the pathophysiology`
- `ening results on the growing share of shopping visits attributed to mobile phones. But, as is the case with engagement, a[[ closer]] look reveals that there`

### 50/50 pools Haiku flags and Opus calls one job

None.

### Real neurons where Opus and Haiku disagree

**70m:real::38**, MLP neuron(s) 54 (network True; labels: haiku C, sonnet C, opus B)

- haiku: `{"label": "C", "patterns": ["LaTeX line-break tokens (\\\\) inside aligned math environments", "the word 'by' after 'denote' or 'indicated' in mathematical prose", "leading whitespace/indentation before the next statement in code", "the 'private' access modifier before a field declaration in Java/C# classes"]}`
- sonnet: `{"label":"C","patterns":["LaTeX line-break \\\\ after \\nonumber in aligned equations (and $$\\ starts)","indentation whitespace token at start of a code line after a block/statement","\"by\" after denote/indicated in math text","\"private\" field modifier in class declarations"]}`
- opus: `{"label": "B", "patterns": ["LaTeX line break (\\\\ or $$) that starts the next line of a multi-line equation, often after \\nonumber", "Leading indentation that starts the next code statement, often after a return/continue block closes", "Repeated 'private' keyword starting the next field declaration in a class", "'by' in mathematical notation phrases such as 'denote by' or 'indicated by'"]}`
- `A_{t+1} = A_t~\}.$$⏎⏎Expanding on the notation of , we shall denote[[ by]] $A_\infty`
- `ihola2013uniform; @kuhlenschmidt2018stability; @Lee2018ccbpf]. We denote[[ by]] $(X^{(n`
- `prime }=$ (\ref⏎{sr2gf}). & ⏎\end{tabular}$$ Above the arrows we[[ have]] put references to the`
- ` || ![EAGLContext setCurrentContext:_context])⏎		{⏎            return nil;⏎		}⏎		⏎	[[	]]_renderer =`

**70m:real::60**, MLP neuron(s) 83 (network False; labels: haiku C, sonnet B, opus B)

- haiku: `{"label": "C", "patterns": ["newline or blank line after a completed statement or line in source code, where a new code block begins", "the auxiliary verb 'was' following a noun-phrase subject in passive scientific prose", "a '+' operator continuing a linear expression in algebra or math"]}`
- sonnet: `{"label":"B","patterns":["Blank-line double newline token between statements in code, inside an indented function body (most snippets)","Token after 'study' in medical abstract ('was') before passive verb","Operator ' +' in math expressions after function call like 'Give 5*a(j)'"]}`
- opus: `{"label": "B", "patterns": ["blank line (double newline) after a statement/block end in code, before the next statement", "occasional ' was' after study subject in abstracts and ' +' in math expressions — clause/segment boundary continuation points"]}`
- `, build_arg_string(kwargs)])⏎                lib.call_module("text", arg_str)[[⏎⏎]]    @fmt_`
- `const Integer code0,⏎    const Integer code1)⏎{⏎    int32 level = start_level;[[⏎⏎]]    while (level`
- `⏎        return⏎    }⏎⏎    let imageViewSnapshot = fromViewTransitionFromView.getViewForTransition()[[⏎⏎]]    let endFrame`
- ` = validRegion⏎        )⏎        assert(sub.requestNext() == expectedMessageWithRegion)⏎    }[[⏎⏎]]    ignore should "`

## pythia-160m

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 15% | 6% | 8% | 12% | 6% |
| network | 49% | 60% | 52% | 48% | 28% |
| haiku | 33% | 62% | 40% | 34% | 16% |
| sonnet | 34% | 56% | 54% | 28% | 16% |
| opus | 34% | 70% | 56% | 36% | 10% |

haiku labels on real neurons: A 57, B 10, C 33

sonnet labels on real neurons: A 5, B 61, C 34

opus labels on real neurons: A 35, B 31, C 34

### 50/50 pools Opus flags and Haiku calls one job

**160m:pooled:50/50:452-1945**, MLP neuron(s) 652, 2930 (network False; labels: haiku A, sonnet B, opus C)

- haiku: `{"label": "A", "patterns": ["token inside a formulaic collocation whose next word is highly predictable, e.g. 'it's [time] to', 'at the [time] of', 'for each [set] of', 'such [that]', 'for the [right] amount of time'"]}`
- sonnet: `{"label":"B","patterns":["\"it's [time]\" to/that, in the construction announcing it's time to do something","noun or adjective right after a determiner or quantifier such as each, either, these, the, or a given (set, end, choices, date, time, right)","scattered sentence-continuation tokens (such [that], paused, Then, so)"]}`
- opus: `{"label": "C", "patterns": ["\"time\" in \"it's time to\" and similar time phrases", "nouns following quantifier/determiner frames (\"for each set\", \"for the right amount\", \"either end\")", "\"such that\" in math", "JSON/code keys (id, url, subscriptionId)", "miscellaneous punctuation and verbs (paused, period)"]}`
- ` change, the extent of conformational change they undergo is comparable to that observed in induced-fit/other interfaces. For both[[ sets]], points corresponding to`
- ` contrast, confidence intervals obtained with the proposed unbiased estimators are shown in Figure \[fig:unlikely:rg\]. For each[[ value]] of $N$,`
- `⏎Kalix nodded in agreement. She herself had been pursued by their agents many times.⏎⏎"Perhaps it's[[ time]] the clan did something`
- `, casting his eyes on the jewel that hung around Kalix's neck, protecting her from detection. "Maybe it's[[ time]] we took that for`

**160m:pooled:50/50:1471-963**, MLP neuron(s) 2219, 1460 (network False; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["a verb or preposition in prose that expects a following object or complement, such as 'subscribing to', 'enroll in', 'refused to', 'prior to joining'"]}`
- sonnet: `{"label": "C", "patterns": ["words about joining, subscribing, enrolling or participating (joining, completing, member, subscribe to, enroll in, participate, sign)", "mid-identifier or repeated-pattern tokens in code and structured text (originalScrollTop, onEndClientData, rake2, brightBlack, Volume, a numeric attribute value, xyz)", "sentence-final punctuation after boilerplate text"]}`
- opus: `{"label": "C", "patterns": ["joining, participating, subscribing, enrolling or membership contexts (e.g. 'joining', 'member', 'subscribe to', 'participate', 'enroll in')", "predictable completions of identifiers or sequences in code and lists (e.g. 'originalScrollTop', 'onEndClientData', 'rake2', 'DIVIDE', 'xyz')", "miscellaneous tokens such as sentence-ending periods, numbers and variables in math or data"]}`
- ` targetThe number of eligible participants required to be screened in order to reach the recruitment target. Key characteristics and reasons for not[[ joining]] the trial will be`
- ` named Christopher, Julian and Talia. He attended Mercer Island High School, where he played high school soccer prior to[[ joining]] the Sounders Academy`
- `.com/understand/Assistants/UAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Tasks/UDaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Fields[[/]]UEaaaaaaaaaaaaaaaaaaaaaaaa`
- ` said India must seek better market access for goods where India has an edge over ASEAN nations, like leather goods and[[ pharmaceutical]]s, to progress`

### 50/50 pools Haiku flags and Opus calls one job

None.

### Real neurons where Opus and Haiku disagree

**160m:real::62**, MLP neuron(s) 92 (network False; labels: haiku A, sonnet B, opus C)

- haiku: `{"label": "A", "patterns": ["token inside formulaic, highly structured text (citations, legal captions, code boilerplate, math problem templates) where the local formatting convention strongly constrains the next token"]}`
- sonnet: `{"label":"B","patterns":["Punctuation/delimiter tokens inside citations and case/reference identifiers (e.g. '(' before year, '-' in docket numbers, '.' after middle initials in names)","Whitespace/indentation or newline tokens in code and license comment blocks","Operators/numbers in synthetic math problems (e.g. '-', ' +', ' 3')"]}`
- opus: `{"label": "C", "patterns": ["punctuation inside citations and reference identifiers (parentheses before years, hyphens and slashes in DOIs, docket and case numbers)", "periods after middle initials in names of judges and lawyers", "whitespace, tab indentation and newlines in source code and license headers", "tokens in arithmetic word problems (answer minus signs, constants in function definitions)"]}`
- `),⏎		"subscriptionId":    autorest.Encode("path", client.SubscriptionID),⏎	[[}]]⏎⏎	const AP`
- ` source and binary forms, with or without⏎// modification, are permitted provided that the following conditions are⏎// met:[[⏎]]//⏎//    `
- ` + b**2 + b + 10. Let z be m(3). Let f(i) = i +[[ 3]]. Calculate z*`
- `07893-5.⏎⏎This work was supported by the European Research Council (Grant Nos. ERC-2012[[-]]StG 30750`

**160m:real::215**, MLP neuron(s) 306 (network False; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["Subword fragment in the middle of a proper name or place name, where the next token completes it (e.g., 'B[ao] Dai', 'K[ual]a Selangor', 'K[olk]ata', 'Sun Y[at]-sen', 'Ky[ushu]')"]}`
- sonnet: `{"label": "C", "patterns": ["mid-word fragments of Asian or foreign proper names (B[ao] Dai, K[ual]a, K[olk]ata, Sun Y[at]-sen, K[ure], [Jun]ichi)", "\"inverse\" before a technical term (inverse Fourier, inverse-gated, inverse Debye mass)", "exponent caret or bracket in scientific notation (10[^]9)", "function words such as \"of\", \"and\", \"is\" or \"to\" in running text", "tab or punctuation tokens in code, lists and numbers"]}`
- opus: `{"label": "C", "patterns": ["subword fragments inside Asian place and person names (Bao Dai, Kuala, Kolkata, Yat-sen, Kyushu, Kure, Junichi)", "exponent markers in scientific notation (10^9, 10^{-10})", "the word 'inverse' in technical contexts", "'of' in fixed phrases (Gulf of, Ministry of, method of, contract of)", "miscellaneous punctuation and formatting tokens (tabs, colons, hyphens in code and numbers)"]}`
- ` come to my house?" "Is that okay?" "Of course it is." "See you." "That's K[[ure]], where the Yam`
- ` temperature heavy quark potential (see Fig. \[fes\](a)). In particular, the screening radius estimated from the[[ inverse]] Debye mass corresponds`
- `reading, and reading their documentation on how to use the crate, I would agree from the surface view. Along with the[[ mult]]ithreading parts of`
- `hed it, oh good,⏎we can tell by the graph that the parabola opens up⏎There's another[[ way]] to tell if the`

## pythia-410m

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 9% | 16% | 12% | 8% | 10% |
| network | 56% | 82% | 64% | 60% | 56% |
| haiku | 46% | 64% | 54% | 54% | 38% |
| sonnet | 38% | 66% | 48% | 44% | 20% |
| opus | 44% | 64% | 68% | 50% | 26% |

haiku labels on real neurons: A 42, B 12, C 46

sonnet labels on real neurons: A 4, B 58, C 38

opus labels on real neurons: A 33, B 23, C 44

### 50/50 pools Opus flags and Haiku calls one job

**410m:pooled:50/50:1495-1976**, MLP neuron(s) 2988, 3955 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["token that completes a highly predictable, formulaic continuation (boilerplate, fixed phrases, idioms, or conventional collocations)"]}`
- sonnet: `{"label": "C", "patterns": ["indentation whitespace at the start of a line in code, before a declaration", "\"those\"/\"that\"/\"some\" as a determiner or pronoun after \"For\", \"if\", or \"But\"", "\"s\" or \"ies\" ending a decade (1970s, seventies)", "\"/\" in the Apache license URL before LICENSE-2.0", "possessive 's and scattered math/LaTeX tokens"]}`
- opus: `{"label": "C", "patterns": ["leading indentation whitespace at the start of code lines", "the 's' plural in decade expressions like 1970s/1960s", "'those'/'that'/'some' as determiners or pronouns in discourse (e.g. 'For those who')", "the '/' after 'licenses' in Apache license URLs", "single-letter variables or indices in math and code expressions"]}`
- `-5.30pm Mar-Oct, to 5pm Nov-Feb) looks for all the world like a 1970[[s]] community centre set in`
- `GET(DmfModule);⏎⏎    // Lock the Module context because the Client Driver may call from different threads ⏎[[    ]]// (e.`
- ` where $F_{n{\bf k}}({\rm i}\tilde{\omega})$ $=$ $\frac{1}{{\rm[[ i]]}\tilde{\omega`
- ` Y_i$ has a density $\frac{\hat f_m g}{\iota_m}{\ensuremath {\mathbb{[[I]]}}}_{I\setminus`

**410m:pooled:50/50:389-923**, MLP neuron(s) 718, 1839 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["token where the next token is strongly predictable from a memorized multi-token phrase, identifier, or name (e.g., 'Never' before 'mind', 'Life' before 'Technologies', '.' inside Java class paths, '_' inside C identifiers)"]}`
- sonnet: `{"label":"C","patterns":["Word-start tokens like \"Never\"/\"never\" (esp. in \"never mind\") and sentence-initial \"There\"","Separators inside identifiers/package names (\".\" in org.apache.commons.math, \"_\" in snake_case names, \"-\" in filenames)","Word fragments \"there/her/read\" that begin compound words (therewith, thereof, ready, readers, her(sker))","\"Life\" in Life Technologies company citations","Closing quote/punctuation tokens after quoted terms"]}`
- opus: `{"label": "C", "patterns": ["dot separator in Java/JS package or namespace paths (org.apache.commons.math.)", "sentence-initial 'Never' / 'There' after sentence boundary", "'there' in legal compounds (therebetween, therewith, thereof)", "underscore in code identifiers", "'Life' in 'Life Technologies' citations", "misc punctuation/quote tokens in code"]}`
- ` mine penge var tabt. De senegalesere, vi havde lært op som optikere[[ her]] i Danmark,`
- `ført.⏎Det vigtigste træk ved det europæiske marked er den fragmentering, der[[ her]]sker på national`
- `.special.BetaTest)⏎testRegularizedBetaPositivePositiveNegative(org.apache.commons.math[[.]]special.Beta`
- `commons.math.util.MathUtilsTest)⏎testMulAndCheck(org.apache.commons.math[[.]]util.MathUtils`

### 50/50 pools Haiku flags and Opus calls one job

None.

### Real neurons where Opus and Haiku disagree

**410m:real::18**, MLP neuron(s) 31 (network True; labels: haiku A, sonnet B, opus C)

- haiku: `{"label": "A", "patterns": ["token is a mid-word subword fragment whose continuation completes a word (e.g., 'osing', 'ing', 'ated', 'ikre')"]}`
- sonnet: `{"label":"B","patterns":["word-internal suffix/subword continuation tokens in the middle of a multi-token word (e.g. -osing, -ating, -ailing, -ered, -ing, -en, Hungarian/German/Danish word fragments)","the word \"last\" (and \"all the\") after a determiner, as in 'the last time', 'the last thing'","code/identifier fragments and hyphenated compound pieces (e.g. __be16, PT_ATTACH, two-stage, low-iron)"]}`
- opus: `{"label": "C", "patterns": ["word-internal continuation pieces completing a partly written word (-osing, -ating, -ailing, -en, Danish ' s' in sikre/sige, Hungarian fragments)", "'last' after 'the' (the last time, the last thing, the last row)", "'the' in the idiom 'all the better/more'", "scattered code and markup tokens (#! /bin/sh, PT_, __be16, path L)"]}`
- `⏎Instagram Lately⏎⏎CT / NY FULL-SERVICE INTERIOR DESIGN⏎⏎FULL-SERV[[ICE]] INTERIOR DESIGN`
- ` more on the exchange). filmed from and followed by Amazon. Please see a friendly UK lead. Please become a college-[[educated]] UK download. H`
- ` Studies in the Physical Sciences_ , 32, 183-196, 2001; M. Gordin, _A Well Or[[dered]] Thing_ , Basic`
- `⏎diesen kleinen Hang hinunter zum⏎echte Skipisten und wenn Sie benutzen⏎dies[[e]] Mülleimer`

**410m:real::65**, MLP neuron(s) 108 (network True; labels: haiku C, sonnet B, opus B)

- haiku: `{"label": "C", "patterns": ["Numeric or variable tokens inside arithmetic and algebra problem expressions (e.g., 'Suppose ... = -4*m + 2*r', 'Expand (...)*(...)')", "Reference and citation markers in scientific paper text (e.g., '[@CR3]', '\\[[@B]5', 'Fig. 2B')", "Mid-word or mid-phrase fragments in prose and names (e.g., 'Yum Y[[um]]', 'creme de la cre[[me]]', 'Slip 'N Sl[[ide]]')"]}`
- sonnet: `{"label":"B","patterns":["Arithmetic/algebra expression tokens in synthetic math problems (operators like + or -, repeated variables or numbers within expressions)","Citation reference markers in scientific text (e.g. [@B3], [@CR3], ref)","Repeated or second half of a duplicated/reduplicated name or phrase (Yum Yum, boo boo, Chop Chop, creme de la creme)"]}`
- opus: `{"label": "B", "patterns": ["Tokens inside arithmetic or algebra expressions in math problems, especially operators (+, -) and terms that mirror or cancel earlier ones (e.g., sqrt(112) - sqrt(112), 3*v, (3 + 1 - 2))", "Tokens that repeat or complete a pattern already seen in context, such as reduplication (Yum Yum, boo boo, creme de la creme) and repeated citation formats ([@B3], [@CR3], ref-type=\"fig\")"]}`
- `4 - 2 - 1) + 3*f + 3*f - 5*f + (-2 - 3 +[[ 3]])*(-2*f`
- `2 - 83*n + 162⏎Expand (3*w**3 + 1 - 1)*(-4 + 1 +[[ 1]])*(4 - 3`
- ` 2*y.⏎-2*y**2⏎Collect the terms in a + 3*a**2 -[[ a]] - 13900 +`
- `dbl>⏎#1 AAPL          -3.92⏎#2 BA             4.29⏎#[[3]] MSFT           3`

## pythia-1b

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 8% | 26% | 18% | 8% | 2% |
| network | 50% | 74% | 68% | 54% | 36% |
| haiku | 47% | 64% | 62% | 52% | 18% |
| sonnet | 41% | 78% | 64% | 58% | 18% |
| opus | 40% | 68% | 72% | 46% | 16% |

haiku labels on real neurons: A 45, B 8, C 47

sonnet labels on real neurons: A 9, B 50, C 41

opus labels on real neurons: A 38, B 22, C 40

### 50/50 pools Opus flags and Haiku calls one job

**1b:pooled:50/50:676-1644**, MLP neuron(s) 2770, 6497 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["Token inside a fixed multi-token expression (phrasal verbs, split words, compounds, identifiers) where the following token is highly predictable"]}`
- sonnet: `{"label": "C", "patterns": ["hyphens or underscores inside compound names and identifiers (L-myc, d-PTFE, uses_from_macos, non_negative)", "verbs before a particle 'out' (figured/figure/pointed out) and adjectives or verbs before a complement ('surprised', 'determined', 'wondered')", "conjunctions or determiners in the middle of a noun phrase (existing and prospective, related and future, one of the following)", "mid-word fragments of names or words (cryptsoft, Regal, Inspired, Municipalities)",`
- opus: `{"label": "C", "patterns": ["mental-state verbs/adjectives after a subject (figured out, surprised, determined, concerned, wondered)", "subword fragments inside names and identifiers (hyphens, underscores, word pieces)", "quantifiers/determiners in phrases like 'of any', 'two', 'multiple', 'following'", "coordinating 'and'/'or' between adjectives or nouns", "object pronouns after 'tell' (tell us, tell me)"]}`
- `towards]. As noted, there may be several ways to define such a temporal structure on systems. We refer to related[[ and]] future work for a`
- `', isa => 'Str', required => 1);⏎⏎  use MooseX::ClassAttribute;⏎⏎  class[[_]]has _api_`
- `⏎The world’s best football stage held only once every four years in Football Carnival Slot Machine World Cup Insp[[ired]] Video Slot Online`
- ` what?'⏎⏎We started walking towards the ponds, and had just reached the first of the enclosures when I[[ figured]] out the source of`

**1b:pooled:50/50:1591-660**, MLP neuron(s) 6290, 2708 (network False; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["Token where the next word or symbol is highly predictable from fixed phrasing, boilerplate, or memorized text, such as 'Thank you', 'Splinter Cell', 'typename' in C++ templates, Wikipedia category lines, court-order wording, and arithmetic or number sequences"]}`
- sonnet: `{"label": "C", "patterns": ["'typename' after 'template<' in C++ code", "polite phrases such as 'Thank you', 'apologies', 'I apologize'", "middle digit chunks inside long numbers or numeric sequences", "'BC' after a century in Wikipedia category lines", "markdown '####' headings", "legal order or judgment wording ('order', 'ORDER', newline after 'Conclusion')", "prepositions or connectives in boilerplate phrases ('questions regarding', 'question for', 'back to')"]}`
- opus: `{"label": "C", "patterns": ["C++ template 'typename' keyword", "digits inside numeric sequences or math expressions", "polite or formulaic phrases (thank you, apologies, questions about/regarding)", "legal or court document structure (order, conclusion, case numbers)", "Markdown '####' headings and category or century labels"]}`
- `]⏎public class RRI_I12_AUTHORIZATION_CONTACT : AbstractGroup {⏎⏎	[[///]]<summary> ⏎`
- `(p).⏎-p**4⏎Let o(k) be the second derivative of k**9/680[[40]] + k**8`
- ` this court.⏎        We ORDER Appellant to SHOW CAUSE in writing within fifteen days of the date of⏎this[[ order]] why this appeal should`
- ` area, it appears as text in the mail⏎From: from email⏎Reply-To: reply email ⏎X[[-]]Mailer: PHP`

### 50/50 pools Haiku flags and Opus calls one job

**1b:pooled:50/50:699-685**, MLP neuron(s) 2829, 2789 (network True; labels: haiku C, sonnet C, opus A)

- haiku: `{"label": "C", "patterns": ["'set'/'set up' used as a phrasal verb meaning arranged or configured", "numeric or identifier tokens inside figure, table, and reference citations", "word-internal subword fragments split mid-word (e.g., 'crick' in cricketing, 'Goe' in Goebel, 'd' in i.d.)", "nouns and determiners following prepositions or articles in narrative prose", "the token right before a noun phrase in dialogue or story text", "mid-sentence tokens in technical or code contexts (e.g., constants`
- sonnet: `{"label": "C", "patterns": ["Token inside or just after a reduced relative clause whose head noun is the gap, e.g. 'the guy I [mentioned]', 'the account you [set] up', 'cameras [set] up', 'the selection mentioned [above]'", "Digit or number tokens in repeated or structured strings, e.g. 'Fig[3]', '0x[10]', 'sqrt([24]48)', 'pone.[01]50345'", "Miscellaneous tokens in the middle of a word or name, e.g. 'Go[eb]el', 'i.[d]', 'their [crick]eting'"]}`
- opus: `{"label": "A", "patterns": ["tokens in modifiers or relative clauses that point back to a previously mentioned or shared referent, e.g. 'the guy I mentioned', 'introduced above', 'that dog you killed', 'the account you set up'"]}`
- ` to the Jordan to find out what all the commotion was about today. I also must admit that the strange man from[[ yesterday]] fascinated me. Again`
- ` the edge of⏎the clearing as they scanned the cabin. One of the men, that same “Ed”⏎who[[ had]] on the previous evening`
- ` order of the circuit court of⏎⏎Stephenson County granting the motion of the defendant, John A.⏎⏎Go[[eb]]el, to dismiss`
- ` domestic violence, filing orders of protection, using shelters, and so on. And yet, despite jumping all the hoops[[ set]] up for them,`

### Real neurons where Opus and Haiku disagree

**1b:real::2**, MLP neuron(s) 10 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["Token where the next token is highly predictable, such as completing a fixed name, phrase, or code idiom (e.g., the 'en' in Kerkkonen, 'son' in Jansson, 'new' before a constructor, '.' before Empty)"]}`
- sonnet: `{"label":"C","patterns":["Code syntax tokens in C#: 'new' keyword, '.' after System/string, 'public/partial/static' modifiers","Mid-word subword pieces of Nordic/Finnish/Swedish surnames and place names (e.g., Toivonen, Jansson, Borås)","Swedish/Finnish language text fragments and 'Swedish' as a language word"]}`
- opus: `{"label": "C", "patterns": ["Subword fragments inside Swedish and Finnish names and words (e.g., -onen, -son, -ström, å, sj, kn)", "C# code syntax tokens, especially 'new' in object creation, plus '.', 'public', 'static', 'partial'"]}`
- `are hade hans dotter⏎där borta och då beslutade huset⏎att min oly[[ck]]a och ingen ville`
- `got är helt ny ingen⏎var som att någon säger ja rätt sa jag⏎nej men de vill[[ sk]]juta`
- `               ?????????????????????????????????⏎             var query = repository.GetAll().Select(x =>[[ new]] Functions⏎            {`
- `1], [@pone.0092177-Toth2], wave action [@pone.0092177-Engkv[[ist]]1], [@pone`

**1b:real::74**, MLP neuron(s) 294 (network False; labels: haiku C, sonnet B, opus B)

- haiku: `{"label": "C", "patterns": ["the token 'file'/'File' in file-handling or filename contexts (e.g., 'Additional file', File.Exists, fileExists)", "mid-word subword fragments completing code identifiers or API names (e.g., ises in assertRaises, get in variable_get, OD in PODS, indexOf)", "the final word of a common noun phrase in prose (e.g., page, paper, curve, appeal, month)"]}`
- sonnet: `{"label":"B","patterns":["the word 'file' or file-related tokens (File, file, Exists, pdf, README, Additional file, article) in code and text","tokens completing a noun/word in a document/page context (message, page, paper, appeal, request, article)","word fragments inside names/identifiers (Es6, Vie jo, HBc Ag, PODS, GSE)"]}`
- opus: `{"label": "B", "patterns": ["Nouns for files, documents and written media (file, page, paper, message, article, pdf, README, 記事), including 'Additional file' and file-API names such as GetFileSystemEntries, File.Exists, fileExists and initWithContentsOfFile", "Mid-word subword pieces inside identifiers and names, such as gene or antigen names (Gag, HBcAg, Nrf2, Hes6, GSE) and surnames (Viejo, Lurie)", "Code tokens for functions or keywords (assertRaises, variable_get, indexOf, IF, form_for, appl`
- ` must be impeccable lest you swerve on the runway—or worse.⏎⏎Related⏎⏎Website of the[[ month]]: BLIPmap`
- ` several factors such as grazing [@pone.0092177-Toth1], [@pone.0092177-V[[ie]]jo2], [@`
- `spb.royalsocietypublishing.org/content/280/1757/20130077.full.[[pdf]]+html). The`
- ` into five open laparotomy groups (n = 10): a control group without vagal manipulation (CON), a bilateral tr[[un]]cal vagotomy with`

## pythia-1.4b

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 7% | 4% | 6% | 2% | 6% |
| network | 49% | 74% | 66% | 42% | 38% |
| haiku | 47% | 58% | 68% | 44% | 36% |
| sonnet | 40% | 66% | 72% | 52% | 26% |
| opus | 47% | 72% | 78% | 50% | 32% |

haiku labels on real neurons: A 45, B 8, C 47

sonnet labels on real neurons: A 5, B 55, C 40

opus labels on real neurons: A 35, B 18, C 47

### 50/50 pools Opus flags and Haiku calls one job

**1.4b:pooled:50/50:650-602**, MLP neuron(s) 2665, 2510 (network True; labels: haiku A, sonnet B, opus C)

- haiku: `{"label": "A", "patterns": ["Word or hyphen inside a set phrase or compound where the next token is highly predictable, such as 'very' before 'little/much/own', 'double' before '-decker/-crosser', 'multi' before '-user/-tool', and 'of' in 'Mistress of the Werewolves'"]}`
- sonnet: `{"label":"B","patterns":["modifier words preceding a noun/compound such as \"very\", \"double\", \"single\", \"multi\" (often before hyphen, e.g. double-decker, multi-user)","\"own/owns\" ownership words","\"of\" in the repeated phrase \"Mistress of the Werewolves\"","formatting/LaTeX/whitespace tokens in code and math"]}`
- opus: `{"label": "C", "patterns": ["number/multiplicity words like double, single, multi- (double-decker, multi-user, single-minded)", "intensifier 'very' (very much, very own, very little)", "'of' in 'Mistress of the Werewolves' and 'owner of'", "ownership verbs (own, owns)", "miscellaneous LaTeX/code formatting tokens"]}`
- ` equation (1):⏎                              θ          beam                =                              sin                          -              1                                ⁡[[                      ]](                                                            `
- ` the team became Midland in ’06. He has since raced in the World Touring Car Championship, part-own[[s]] the Ocean GP2`
- `⏎Cause it's a warrior you seein' here tonight in the flesh⏎I give you somthin' high voltage[[ double]] dosage⏎Runn`
- `ism Speaks goal!⏎$10 suggested donation / $5 for raffle tix⏎⏎May 7 @[[ Double]] Door⏎It's`

**1.4b:pooled:50/50:164-71**, MLP neuron(s) 640, 295 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["Mid-sentence pause points where the text breaks off or turns, such as dashes, ellipses, or hedge words like 'sure', 'clear', and 'much' before 'rather', and the next words continue or qualify the clause"]}`
- sonnet: `{"label": "C", "patterns": ["\" sure\" / \"Sure\" as an interjection or in \"you're sure\", \"sure enough\", \"Surely\"", "em-dash or ellipsis that interrupts a sentence and continues it", "\" clear\" or \" much\" after a copula or \"would\" (\"would much rather\")", "word-fragment tokens such as \"Pres\" and \"eas\" at the start of a longer word", "math exponent markers \"**\" and \")**\" in generated arithmetic", "markdown citation fragments such as \"](#\" and \"^](#\""]}`
- opus: `{"label": "C", "patterns": ["certainty/assertion words such as 'sure', 'Sure', 'clear' and 'presumably'", "'much' in 'would much rather'", "em-dashes and ellipses that interrupt or continue a clause", "markdown citation link fragments like '](#'", "exponent '**' and closing ')**' in math expressions", "word-start fragments such as 'eas' and 'Pres', plus newlines and whitespace"]}`
- ` 315 = 387. Suppose c - 9*c + l = 0. Let h(o) = o + 9[[.]] Calculate the least common`
- ` briefcase." "But I couldn't swear about brown." " He came out of the hotel?" " Yeah." "[[Sure]], that figures."`
- ` for cosmic-ray background) and with the exposure rates measured by total gamma-ray counting. From this and other results[[...]]⏎⏎An X`
- ` similar fuel specification), except for the absence of lead. Unofficially, in some aviation fuels industry circles, such usage[[—]]i.e.`

### 50/50 pools Haiku flags and Opus calls one job

**1.4b:pooled:50/50:380-1332**, MLP neuron(s) 1613, 5467 (network True; labels: haiku C, sonnet B, opus A)

- haiku: `{"label": "C", "patterns": ["months and dates in court filing or document headers (e.g., \"December 30, 2003\", \"March 10, 1971\")", "the '#' that begins a C preprocessor #define macro line", "numeric tokens in probability or arithmetic expressions (e.g., \"[[3]]/340\", \"[[63]]/168\")", "article-number fragments in journal DOIs (e.g., \"fphar-[[09]]-00166\")", "mid-sentence function words and verbs in news and web prose (e.g., \"cannot\", \"provided by\", \"is [[provided]]\", \"do not\")"]}`
- sonnet: `{"label":"B","patterns":["month or date tokens in legal case headers and 'As of …' phrases","time-frequency words such as each, every, year, per and Thursday","'#' before 'define' in C headers, and numeric tokens in repeated templates (fphar-09, 3/340)","auxiliary or modal words in boilerplate disclaimers (cannot, do, necessarily, is, are, only)"]}`
- opus: `{"label": "A", "patterns": ["tokens inside formulaic, boilerplate or templated text where the continuation is highly predictable, such as legal headers and dates, disclaimers, citation IDs, #define lines and stock phrases"]}`
- ` city located in Osaka Prefecture, Japan. The city was founded on September 1, 1956.⏎⏎As of[[ March]] 2017, the city`
- `#^*P* \< 0.05 and ^\#\#^*P* \< 0.01).](fphar-[[10]]-00552-`
- ` that company.⏎⏎Whilst Niche Jobs Ltd makes every effort to restrict CV access to legitimate companies only, it[[ cannot]] be held responsible for`
- ` national selection committee made up of 24 experts, including toy collectors, designers and psychologists vote the winners in to the hall each[[ year]]. Anyone can nom`

### Real neurons where Opus and Haiku disagree

**1.4b:real::26**, MLP neuron(s) 90 (network True; labels: haiku C, sonnet C, opus B)

- haiku: `{"label": "C", "patterns": ["the word \"back\" used as a body part or physical location (the back of the neck, lay back, back yard)", "abbreviation or rule-name tokens in legal citations, especially the \"R\" in Fed. R. App. P. / Fed. R. Civ. P. and similar court rules", "mid-word or mid-name fragments that complete a term, such as \"retin\"opathy, \"ular\" degeneration, and \"hyper\"activity", "capitalized name or abbreviation starts inside organization or proper-noun names, such as \"H\"osp., `
- sonnet: `{"label": "C", "patterns": ["the word \"back\" (back of, lay back, back and forth, backyard)", "mid-term tokens in legal citations and abbreviations (Fed. R., U.S., Hosp., F.C.)", "word-fragment or proper-name pieces in medical and other terms (macular, retin|opathy, Aquinas)"]}`
- opus: `{"label": "B", "patterns": ["tokens partway through fixed multi-word names, terms and abbreviations, e.g. 'Fed. R.', 'U.S.', 'macular degeneration', 'diabetic retinopathy', 'Adverse Events', 'Hosp.', 'F.C.'", "the word 'back' in set phrases, e.g. 'back of', 'back and forth', 'lay back', 'his back', 'BACKGROUND'"]}`
- ` were also cleared to return, but again, Scarlett is the most consequential of the group. The St. Thomas[[ A]]quinas alumn`
- ` 30 patients compared combination therapy of ranibizumab with PRP vs. PRP alone in treatment-naïve proliferative diabetic[[ retin]]opathy (PDR`
- `, and toxicity assessment were performed on a weekly basis. Acute toxicity was graded according to Common Terminology Criteria for Ad[[verse]] Events (CTCA`
- `.C. players⏎Category:Maidenhead United F.C. players⏎Category:Thamesmead Town[[ F]].C. players`

**1.4b:real::132**, MLP neuron(s) 529 (network False; labels: haiku A, sonnet B, opus C)

- haiku: `{"label": "A", "patterns": ["Function words (prepositions, infinitival 'to', 'be', 'such', 'I', 'where', 'who') in mid-phrase positions that open a grammatical complement, so the next token is tightly constrained by syntax"]}`
- sonnet: `{"label": "B", "patterns": ["function words that begin a postmodifier or complement after a noun or verb (\"of\", \"to be\", \"such (as)\", \"where\", \"I\", \"to\", e.g. \"copies of\", \"to be determined\", \"things I want\")", "mid-word fragments inside rare names or words (Kh[od]nev, G[urd]urr, W[ond]olowski, Ç[or]bası, [bur]gundy)", "scattered other tokens in the middle of a phrase where the continuation is predictable (\"known\", \"after\", \"used\")"]}`
- opus: `{"label": "C", "patterns": ["tokens that begin a post-nominal modifier or complement after a noun, such as 'of' in 'pile of', 'to be' in 'inferences to be drawn', 'I' in 'things I want', 'where' in 'area where'", "word-internal subword fragments inside rare names or foreign words, such as Kh-od-nev, G-urd-urr, W-ond-olowski, Ç-or-bası", "scattered unrelated tokens such as newlines after headings and 'It' or 'I' at the start of a clause"]}`
- `, in determining whether there is a genuine issue of material fact, the Court views the facts, and all reasonable inferences to[[ be]] drawn from them,`
- `'s see, how has this week gone. It has been long and a bit on the warm side but that is to[[ be]] expected since it is`
- ` who present with new neurologic deficits \[[@B6]\]. Another reason to obtain an INR on admission in these patients may[[ be]] its ability to help`
- `arly Musonda, who each provided a more attacking threat. Alex Kiwomya and Reece Mitchell once again flank[[ed]] Abraham in attack,`

## pythia-2.8b

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 3% | 10% | 4% | 8% | 4% |
| network | 44% | 50% | 54% | 50% | 40% |
| haiku | 46% | 78% | 58% | 46% | 30% |
| sonnet | 52% | 64% | 74% | 50% | 24% |
| opus | 54% | 80% | 72% | 48% | 24% |

haiku labels on real neurons: A 42, B 12, C 46

sonnet labels on real neurons: A 4, B 44, C 52

opus labels on real neurons: A 27, B 19, C 54

### 50/50 pools Opus flags and Haiku calls one job

**2.8b:pooled:50/50:1310-1307**, MLP neuron(s) 6607, 6580 (network True; labels: haiku A, sonnet C, opus C)

- haiku: `{"label": "A", "patterns": ["Mid-sentence tokens in running English prose, often verbs or function words (of, for, with, in, own, said, told) where the phrase continues into a verb or object"]}`
- sonnet: `{"label":"C","patterns":["finite verbs after a subject or relative pronoun (said, told, owns, provide, sell, has)","'of', 'own', 'even' and other function words or modifiers mid-phrase after a quantifier or possessive (much of, your own, more of)","numerals and operators inside chemical names, math expressions and section numbers"]}`
- opus: `{"label": "C", "patterns": ["main verb right after its subject, especially in relative clauses ('that owns', 'which owns', 'that provide') and speech attributions ('McMahon said', 'Gorka told')", "'of' and determiners in quantity or partitive phrases ('much of', 'six months of', 'more and more of', 'all the')", "digits and symbols inside chemical names, math expressions and section numbering"]}`
- `:⏎⏎Blasphemy against Allah: Death⏎⏎Blasphemy against the Prophet Muhammad: Death –[[ even]] if the accuser`
- `*v**3 + 10*v**2 - 1 wrt v?⏎-1788*f**2 -[[ 24]]*f + 24`
- ` requests for Content Objects, and routers across the network route these requests directly through the network to a closest network node that[[ stores]] a copy of the`
- ` with comfy wooden bunks, a convivial living room with peat fire and a welcoming owner who can provide[[ all]] kinds of advice on`

**2.8b:pooled:50/50:1414-1067**, MLP neuron(s) 7165, 5364 (network False; labels: haiku A, sonnet B, opus C)

- haiku: `{"label": "A", "patterns": ["Mid-sentence tokens in running prose or formal text, usually function words, partial words, or line/paragraph breaks, where the sentence or citation is still open and the next token continues the clause"]}`
- sonnet: `{"label":"B","patterns":["token right before a proper name or case name, such as 'in' before a case name, or 'member', 'group', 'Chairman' or 'Protect' before a person or organization name","newline or paragraph break before a new section or heading, often in legal or citation text","punctuation or closing parentheses in legal citations","function words in comparative or clause structures, such as 'the' in 'the longer', 'because', 'does', and 'been'"]}`
- opus: `{"label": "C", "patterns": ["legal citation and case-reference context (court opinions, 'in [case]', citation punctuation)", "newlines and section breaks between text blocks", "miscellaneous tokens in general prose and code with no common feature"]}`
- ` reconsideration of this Court’s June 27, 2017⏎  order is considered, and it is DENIED, because it[[ does]] not appear that the`
- ` Me and her are tight so I know she will get it and be reading it.⏎⏎Let's see, how[[ has]] this week gone.`
- ` 41, 45-46, 78 S.Ct. 99, 2 L.Ed.2d 80 (1957),[[ that]] "a complaint should`
- `iorari to the United States Court of Appeals for the Fifth Circuit.⏎The petition for writ of certiorari is denied.[[⏎]]Justice BRENNA`

### 50/50 pools Haiku flags and Opus calls one job

**2.8b:pooled:50/50:1413-768**, MLP neuron(s) 7161, 3799 (network False; labels: haiku C, sonnet C, opus A)

- haiku: `{"label": "C", "patterns": ["numeric values inside legal case citations (reporter volume and page numbers)", "word after an infinitive or verb-complement 'to' (e.g., 'to [want]', 'to [request]')", "code syntax tokens (URLs, operators, method parentheses, loop increments)", "next term or value in numeric sequences and algebra expressions", "function-word choices in general prose (e.g., 'of', 'if', 'that', 'what')"]}`
- sonnet: `{"label":"C","patterns":["digits in legal citations (L.Ed.2d 611, S.Ct. numbers) and in numeric sequences or algebra expressions","function words or verb complements in prose where a specific continuation is expected (to, of, if, for, want, that)","references like 'previous theorem/Lemma/proposition' and LaTeX or code punctuation (subscripts, parentheses, quotes)"]}`
- opus: `{"label": "A", "patterns": ["tokens that are highly predictable from the preceding context, such as fixed phrases ('whether or not', 'at least', 'field of view'), arithmetic sequence continuations, legal citation page numbers, and code or LaTeX syntax completions"]}`
- `能使用2GB的内存，而且程序占用的内存越大，�[[�]]址效`
- ` + 12⏎What is the h'th term of 5728, 5727, 5726, 5725, 57[[24]], 5723?`
- `⏎in Rosamond?⏎You can browse all Rosamond real estate or⏎contact a Coldwell Banker agent to[[ request]] more information.<|endoftext|>`
- `, no one took the seat next to her. Kalix looked too mad, dirty, and altogether strange for anyone to[[ want]] to spend five hours`

### Real neurons where Opus and Haiku disagree

**2.8b:real::189**, MLP neuron(s) 868 (network False; labels: haiku A, sonnet B, opus C)

- haiku: `{"label": "A", "patterns": ["Token right before an expected grammatical complement, object, or argument (e.g., a verb or noun phrase to follow), across prose and code"]}`
- sonnet: `{"label":"B","patterns":["transitive verbs or participles awaiting an object (recognize, heard, considered, named, protect, admit, thank)","prepositions or 'to' awaiting a noun phrase (fair to, post for, up to, for migrants)","colons or opening brackets introducing code or list content (look like:, setValue([, s.add([)","modifier or name fragments awaiting a noun (cruel and unusual, Modified Eagle, known facts, additional)"]}`
- opus: `{"label": "C", "patterns": ["verbs/nouns completing predicates in formal legal or academic prose (recognize, heard, account, honor)", "code punctuation tokens such as '([', '[]', ':' before a code block", "units and symbols in scientific notation (μ*M, PCR, Eagle's medium)", "miscellaneous function words (to, for, last)"]}`
- ` use the Latin form of their names), a brilliant physician who supposedly spoiled his legacy by insisting in the statutes that the college[[ admit]] no 'deaf`
- ` As we have said, at his trial, defendant had stated he had been asked to come to the victim's home to[[ see]] if he could get`
- ` it makes me more cognisant of these issues when I travel. We were with a high school group outside the US[[ last]] year, and every`
- ` slaves too and still do …⏎⏎Joseph⏎⏎If these benevolent black slave owners kept black slaves “to[[ protect]] that person”, what`

**2.8b:real::202**, MLP neuron(s) 930 (network True; labels: haiku B, sonnet C, opus C)

- haiku: `{"label": "B", "patterns": ["Mid-word subword fragments that are completed by following tokens (e.g., 'ots' in Botswana/Cotswolds, 'eps' in Stellar eps-matches, 'ps' in dyspepsia, 'ars' in arsB)", "The particle 'out' after a verb or motion phrase (e.g., 'pull out of', 'headed out', 'radiating out from')"]}`
- sonnet: `{"label":"C","patterns":["mid-word subword fragments like 'ots' in Botswana/Cotswolds, 'eps', 'ps', 'ars', 'ops', 'aps' tokens","the word 'out' after verbs (pull out, radiating out, out there)","LaTeX '{\\' tokens after \\frac and '}{\\'"]}`
- opus: `{"label": "C", "patterns": ["Word-internal subword fragments, mostly vowel+consonant pieces ending in -s ('ots' in Botswana/Cotswolds, 'eps' in eps-matches/catalepsy, 'ps', 'ars', 'ops', 'ols', 'aps', 'ens', 'urs', 'am', 'iam')", "The particle ' out' after a verb (headed out, get it out, pull out, radiating out)", "LaTeX fraction brace tokens such as '{\\' after \\frac, or ')}{'"]}`
- ` moderately active (2, 3, 4, and 6), and inactive (1, 7, and 8). The catal[[eps]]y-reversal`
- ` pH monitoring, an abnormal score does not appear to predict response to proton pump inhibitor therapy in patients with unspecified functional dyspe[[ps]]ia.<|endoftext|>The`
- `\big((I\setminus[0,\varepsilon_m])\cap {\ensuremath {\mathbb{R}}}_+\big[[)}{]]m-1},\`
- `M2I10M;mutations=58A,59C,61T⏎seq1	Stellar	[[eps]]-matches	99`
