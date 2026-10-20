# v11 results per model

Flag rates (unrelated mixed) by reader and unit type at the quarter and three-quarters depths, then, per model, up to 2 real neurons each reader alone flags and 2 50/50 pools the language model calls one job, with its stated patterns and four of the unit's 50 snippets (its 1st, 2nd, 26th and 27th strongest; the language model saw all 50 in random order). Unit ids use v6's sample indices; the MLP neuron ids are given beside them. The middle depth is in [v7_models.md](v7_models.md).

## quarter

### pythia-70m, quarter

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 10% | 26% | 12% | 10% | 14% |
| network | 50% | 66% | 72% | 54% | 40% |
| llm | 28% | 60% | 60% | 34% | 30% |

Language-model labels: {'real': {'B': 13, 'A': 59, 'C': 28}, 'pooled 50/50': {'C': 30, 'B': 4, 'A': 16}, 'pooled 70/30': {'C': 30, 'A': 17, 'B': 3}, 'pooled 90/10': {'C': 17, 'B': 11, 'A': 22}, 'related': {'A': 28, 'C': 15, 'B': 7}}; without a label: 0

#### Real neurons only the language model flags

**70m:quarter:real::351**, MLP neuron(s) 358 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["\"up\" (and \"out\") as a verb particle in phrasal verbs such as picked up, sat up, filled out", "\"U\" as the volume abbreviation in U.S. Reports legal citations", "\"vegetarian\" in food and lodging descriptions", "miscellaneous single-token cases (\"pm\", \"liable\", \"SYSTEM\", \"released\", \"ok\", \"raw\", \"performance\", \"premiered\", \"ix\")"]}`
- ` held me up—it was fine work——”⏎⏎“Oh, shut your blamed mouth!” growled Tom, sitting[[ up]]. “I got`
- ` anybody that had a beef with him?" "Naw, everybody loved him and if they didn't, they'd sit[[ out]] of his way."`
- ` the author knows what he is writing about, that he has⏎been right down there himself, and has not just sat[[ up]] in the grand⏎`
- ` that it was hopeless, it was too weak to survive. It would not even last the journey home. As I picked[[ up]] the rock I thought`

**70m:quarter:real::389**, MLP neuron(s) 397 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["the word 'error' in legal appellate contexts (e.g., 'no error', 'assignment of error', 'plain error')", "'size' in study or sample descriptions (e.g., 'sample size')", "'iso' as a word-piece in chemical or enzyme names (e.g., 'isozymes', 'iso-C16')", "'transfer' in 'methyltransferases'", "'character' in legal or statutory factor lists (e.g., 'nature and character')", "'support' meaning evidentiary backing (e.g., 'evidence to support it')"]}`
- ` April 21 for a gun permit and had a restraining order against Eitel at the time of her death.⏎⏎Pro[[test]]ors blame gun control`
- ` one way or the other concerning the guilt or innocence of the defendants. The questioning of witnesses by the court did not constitute[[ error]]. Moore v.`
- ` with the null hypotheses that residuals are minimal and the data fit the model well). The Chi-square is influenced by sample[[ size]], which can lead`
- `95% confidence interval; HbA1c = glycated hemoglobin.⏎⏎^a^N is not[[ equal]] to 20 for all`

#### Real neurons only the network reader flags

**70m:quarter:real::30**, MLP neuron(s) 31 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["word right after a coordinating conjunction or 'not' that opens a parallel prepositional or adverbial phrase (e.g., 'and [by]', 'or [with]', 'and [at] least')", "'out' in 'N out of M' counts and 'out-of' constructions", "mid-word subword fragments that continue a split word (e.g., 'et' in etched, 'EL' in ELISA, 'aden' in adenine)"]}`
- `layer, etching the exposed tri-layer, removing the mask, depositing a layer of polyimide over the un-[[et]]ched copper, forming`
- `uttered when the ions strike the surface. As a result, dielectric material deposited on the wafer surface is simultaneously sputter-[[et]]ched to help keep`
- ` RNA genome is 10--11 kilo-bases in length, bearing a type 1 cap and lacking a poly-[[aden]]ine tail. It`
- ` may be explained by technical differences in nucleotide measurement \[[@CR28]\] and subcellular AMP compartmentalisation \[[@CR29]\], or[[ by]] the fact that immortal`

**70m:quarter:real::50**, MLP neuron(s) 51 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token where the next token is highly predictable, completing a split word or fixed collocation (e.g., 'No [one]', 'Vo[ices]', 'Lonely [Planet]', 'Rainforest [Action]')"]}`
- ` all across Russia." "This room will be alive again...with happy voices and the patter of little feet." "Vo[[ices]] from across Europe being`
- ` grande forêt, où il fit la rencontre de bandits qui s'apprêtaient à déval[[iser]] le trésor`
- ` en Boca Juniors no quieren dejar pasar otro año más en blanco internacionalmente. Per[[der]] la final del Bern`
- ` estreiam, as empresas já investiram uma grande quantia de dinheiro para transformá-los[[ no]] que são", cont`

#### 50/50 pools the language model calls one job

**70m:quarter:pooled:50/50:538-956**, MLP neuron(s) 549, 975 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-word or mid-phrase tokens inside a fixed multi-token expression (names, legal citations, product names, idioms) whose next token is highly predictable from the surrounding fragment"]}`
- ` exhaustive and is not intended to single out any particular use as presumptively fair.  Harper & Row, 471[[ U]].S. at`
- `ailed to the floor. What had happened to Terence? All his charm and sophistication had disappeared, leaving only cr[[ud]]eness behind. She`
- `-In-Progress]⏎⏎These are all Sonic Mania Plus cheats discovered so far on Switch, Xbox[[ One]], PS4 &`
- ` God is easy. What’s harder – and much more important – is creating a world that no longer has need of[[ Him]].⏎⏎Like`

**70m:quarter:pooled:50/50:1092-1040**, MLP neuron(s) 1119, 1064 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Word-internal subword fragment in the middle of a split word, where the unit fires on a piece that the model must complete into a longer word (e.g., 'okes' in Crookes/evokes, 'vents' in circumvents, 'aliqu' in aliquots)"]}`
- ` been demonstrated in the preexisting literature of other organisms as being involved either in Si transport or in Si polymerization. This circum[[vents]] the need to carry`
- `^* [@pone.0008549-Toivonen2], breakdown of fatty acids by beta-oxidation partially circum[[vents]] the problem by feeding`
- ` was founded to encourage billionaires to donate half of their wealth to charity and includes other famous philanthropists like Bill and[[ Mel]]inda Gates and Warren`
- `agált: "Tájékoztatásul közlöm, hogy angliai m[[unk]]ámat tö`

### pythia-160m, quarter

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 10% | 6% | 12% | 14% | 10% |
| network | 55% | 72% | 62% | 52% | 38% |
| llm | 38% | 56% | 54% | 34% | 22% |

Language-model labels: {'real': {'A': 48, 'C': 38, 'B': 14}, 'pooled 50/50': {'A': 19, 'C': 28, 'B': 3}, 'pooled 70/30': {'C': 27, 'A': 16, 'B': 7}, 'pooled 90/10': {'B': 9, 'C': 17, 'A': 24}, 'related': {'B': 6, 'A': 33, 'C': 11}}; without a label: 0

#### Real neurons only the language model flags

**160m:quarter:real::172**, MLP neuron(s) 251 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["the \"%\" sign following a numeric concentration or percentage, mainly in lab protocols", "word-internal subword fragments completing a split word or name (e.g., ZE[[ISS]], Bchw[[el]]ger, beschre[[ib]]bar)", "\"most\" or \"an\" in fixed phrases like \"for the most part\" and \"such an\"", "\"election\" completing \"re-\" in political text"]}`
- `mm × 300 µm i.d.; Thermofisher) for 1 min with 0.1[[%]] TFA at a`
- ` distilled water. Sections were stained with hematoxylin phloxin safran (HPS) and with 0.1[[%]] alcian blue (`
- ` 10-second blast of air applied to the vital tooth was scored pre-operatively and again within a one-to-[[four]] week post-ce`
- `ires are of limited interest to me aside from on a technical level. I do, however, have a huge mancr[[ush]] on the e-`

**160m:quarter:real::467**, MLP neuron(s) 698 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["LaTeX/math punctuation and delimiters (commas, brackets, operators, and closing braces in formulas)", "Token immediately after a proper name, title, or named entity in prose, where a descriptor or qualifier is expected next", "Punctuation and separators in code, URLs, and file paths"]}`
- `,\chi}^{\chi,\chi}=-\sum_{\chi'\neq\chi}W_{\chi',\chi}^{\chi',\chi[[}$]] in order to ensure`
- ` and Environmental Art Research Centre. ⏎⏎In 2011 "I am a Fantasy", was exhibited at the PayneShurvell[[ gallery]] in East London from`
- `ates a number as defined by the CSS spec.⏎ */⏎class HTMLPurifier_AttrDef_CSS_Number[[ extends]] HTMLPurifier_`
- `'_{\lambda'}(t,q)= c_{\lambda}(q,t) c'_{\lambda}(q,t[[),]] \\⏎&\prod`

#### Real neurons only the network reader flags

**160m:quarter:real::179**, MLP neuron(s) 262 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Marked token completes a word split across subword tokens (mid-word fragment continuation, e.g. 'Wes[[sex]]', 'fin[[i]]', 'halorhodops[[in]]')"]}`
- `'ai dit non mais il veut tout faire⏎hologramme sur vous, il a dit que ce ser[[ait]]⏎être des CD`
- ` in _bel puris_ , spiced lamb _chapli kebabs_ , _papri cha[[at]]_ and railway lamb`
- `CGT\]a) that many thin-flake [Cr$_{2}$Ge$_{2}$Te$_{6}$]{}[[ samples]] can be produced through`
- `rangle_{\text{2}},k_{\text{2}})]$$ To account for NBDs not describing the 0[[–]]bin and the first`

**160m:quarter:real::261**, MLP neuron(s) 383 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["apostrophe in a possessive before 's' in proper names (e.g., Dulbecco's, Pfirrmann's, Student's)", "period or delimiter inside figure-file identifiers and citation fragments (e.g., pone.0225795.g025, F.Supp.)", "line break at the end of license headers, court-case captions, or copyright boilerplate"]}`
- `2ws) (TR, 5000 milliseconds; TE, 111 milliseconds).⏎⏎Images were analysed by Pfirrmann[[\']]s grading \[[@B`
- `, [@B25]), DRT (García-Madruga et al., [@B29]) and Raven[[\']]s Progressive Matrices`
- `ca13044-sec-0005}⏎--------------------⏎⏎*P* \< 0.05 indicated statistical significance. Fisher[[\']]s exact tests were`
- ` in a book review.");  Robert Stigwood Group Ltd. v. O'Reilly, 346 F.Supp[[.]] 376, 385 (`

#### 50/50 pools the language model calls one job

**160m:quarter:pooled:50/50:1947-1365**, MLP neuron(s) 2924, 2022 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token inside a fixed multi-token word or idiom whose continuation is highly predictable (e.g., 'of' in 'of course', 'ill' in 'scintillator', 'IN' in 'LEGUMINOSAE')"]}`
- `Arts and Crafts' over factory-made materials. He even dreamt up the wallpaper designs.⏎⏎[[Of]] particular note are Rus`
- ` requests poured in to DI President, Dr Widad Akrawi, on twitter and other social networks.⏎⏎What[[ can]] be saidWords disappear`
- ` and under the quirky Mathematical Bridge. You'll soon wonder how you could have studied anywhere else.⏎⏎PHIL[[IP]] GAME / GET`
- ` studied elsewhere and involve the topological-momenton generators $(17)-(20)$.⏎⏎THE ELECTRON - QUAS[[IP]]ARTICLE TR`

**160m:quarter:pooled:50/50:687-326**, MLP neuron(s) 1014, 481 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token that completes a highly predictable formulaic phrase or collocation, such as 'In doing [so]', 'say [nothing]', 'wild[fire]', 'hi[atus]', and 'a world [where]/[of]'", "token predicted by a stock biographical or encyclopedic frame, such as 'is a [former]' before a nationality or profession, and day-of-month after a month name"]}`
- ` development of two inner electron rings."⏎⏎Bohr soon returned to his original claim about element 72. In doing[[ so]], he provided further`
- ` that without it, AU would be much, much weaker than it is today. And I don't blame anyone in the[[ slightest]], for chasing it`
- ` our goalposts. Terms like "significantly" and "radically" are relative. If we lived in a world[[ where]] our only basis for`
- `, and, often being irrevocable, it has been a source of understandable⏎⏎concern in a society[[ where]] couples who may already`

### pythia-410m, quarter

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 14% | 12% | 16% | 2% | 8% |
| network | 49% | 68% | 70% | 50% | 42% |
| llm | 34% | 46% | 50% | 34% | 16% |

Language-model labels: {'real': {'A': 56, 'C': 34, 'B': 10}, 'pooled 50/50': {'A': 18, 'C': 23, 'B': 9}, 'pooled 70/30': {'C': 25, 'A': 16, 'B': 9}, 'pooled 90/10': {'C': 17, 'B': 8, 'A': 25}, 'related': {'A': 35, 'C': 8, 'B': 7}}; without a label: 0

#### Real neurons only the language model flags

**410m:quarter:real::22**, MLP neuron(s) 56 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Legal case captions and court-document headers, where the token falls at a line break or party designation before a court name or case number", "Legal actors and roles such as prosecutor, victim, counsel, and superintendent in court-opinion text", "Empty braces '{}' completing a C++ constructor or member initializer", "Python interactive prompt '>>>' starting a new REPL input line"]}`
- `ASMUSSEN, Respondent,⏎v.⏎FRESNO TRACTION COMPANY (a Corporation) et al., Appellants[[.]]⏎Civ. No`
- `IATION, Plaintiff and Respondent,⏎v.⏎ROYAL DUTCH DAIRIES, Defendant and Appellant[[.]]⏎Civ. No`
- `igende grad blive anerkendt på forsvarsområdet og inden for vores uden[[rig]]spolitik.`
- ` CHECK: alloca i64, align 8⏎⏎⏎// Make sure we honor the aligned attribute.⏎struct X {[[ int]] x,y,`

**410m:quarter:real::76**, MLP neuron(s) 169 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["'solution(s)' or 'solution method' in optimization and mathematics contexts (snippets 1, 5, 10, 12, 15, 23, 29, 32, 37, 42, 50)", "'review' or 'requirement' in legal standard-of-review and statutory-test phrasing (snippets 3, 8, 16, 19, 20)", "apostrophe or possessive 's' completing a title or proper noun (snippets 17, 18, 24, 33, 34)"]}`
- ` Guide)⏎⏎C# lets you instantiate an object or collection and perform member⏎  assignments in a single[[ statement]].⏎⏎Collection`
- `prevailing party. Because most PCRA appeals involve questions⏎       of fact and law, we employ a mixed standard of[[ review]]. We defer⏎`
- ` heard before. And think about how well this fits with frugality and financial independence!⏎⏎The Monk and[[ the]] Minister⏎⏎Two`
- ` itself – afloat in a protected dock – is the star attraction. You can visit the bridge, the galley and[[ the]] mahogany-`

#### Real neurons only the network reader flags

**410m:quarter:real::101**, MLP neuron(s) 221 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token inside a fixed multi-token term, date, or word fragment where the next token is strongly determined by the preceding characters"]}`
- ` in question or that the people stopped are the best measure of who is eligible to be stopped. If racial prejudice were af[[oot]], the standard would`
- ` of Appeal. It was published in 1996. Much of the report was secret.⏎⏎Background⏎In the late 1980[[s]], Matrix Churchill,`
- ` order was the opposite. The three tests were presented to students in specific booklets. Prior to data collection, permission was[[ obtained]] from parents, teachers`
- ` de interés y le negó el último favor a Alberto Fernández, que promete bajarl[[as]]⏎⏎BlackFriday`

**410m:quarter:real::135**, MLP neuron(s) 288 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Promotional web copy where the marked token sits just before a call to action, such as subscribe, register, click, contact, visit, download, or select"]}`
- ` being a mother to you?⏎⏎Want More?⏎⏎Have our best reads delivered straight to your inbox every week[[ by]] subscribing to our`
- ` social account or click here to log in I would like to receive lunchtime headlines Monday - Friday plus breaking news alerts,[[ by]] email Update newsletter preferences`
- ` you're human:⏎⏎November's theme is EVOLUTION. For details on how to send us your work,[[ please]] read our Submit`
- ` a good indicator of market activity, it provides little information for targeting wood building systems to specific markets.⏎⏎Click on[[ images]] to enlarge⏎⏎`

#### 50/50 pools the language model calls one job

**410m:quarter:pooled:50/50:36-140**, MLP neuron(s) 83, 297 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Token immediately before a noun phrase that gives the category or type of the subject, especially the article or verb right after \"is\" or \"be\" in mathematical and encyclopedic definitions (e.g., \"Let X be [a] Stein manifold\", \"Spazigaster is [a] genus\")"]}`
- ` be n(0). Suppose v*x - 15*x - 16 = 0. Calculate the remainder when 44 -[[ (]]39/15 -`
- `+2r) \times Sp(2n))$ – type C II⏎-------------------------------------------------⏎⏎Let $r$ be[[ a]] non-negative integer`
- ` proto, že⏎pokud si pamatujete, dříve měla jeho dcer[[a]]⏎tam a`
- `://localhost:3306/datomic?user=datomic&password=datomic")⏎(d/create[[-]]database uri)⏎`

**410m:quarter:pooled:50/50:1249-178**, MLP neuron(s) 2504, 385 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Tokens completing formulaic stock phrases and collocations, such as 'as mentioned/noted/discussed above', 'dotted/lined/come with', 'you might say', 'where you'll also find', and 'there will be'"]}`
- ` hands of a tanker who can fire its 75 mm cannon accurately while on the move. All Cromwell B bundles come[[ with]] a "zero-`
- `ce Bookstore on Main Street, where I will be part of the SinC Central Coast Chapter book event. There will[[ be]] 8 local authors of`
- ` look at), I'd like to look at it, but otherwise I'm a bit skeptical.⏎⏎As I noted[[ above]], the premise of`
- `.[5]⏎⏎2. Failure to Plead Fraud Claim with Requisite Particularity⏎⏎As mentioned[[ above]], Kennedy also argues`

### pythia-1b, quarter

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 13% | 8% | 16% | 18% | 0% |
| network | 52% | 52% | 66% | 42% | 46% |
| llm | 38% | 40% | 42% | 40% | 20% |

Language-model labels: {'real': {'C': 38, 'B': 10, 'A': 52}, 'pooled 50/50': {'C': 20, 'A': 19, 'B': 11}, 'pooled 70/30': {'A': 22, 'C': 21, 'B': 7}, 'pooled 90/10': {'C': 20, 'A': 27, 'B': 3}, 'related': {'A': 34, 'C': 10, 'B': 6}}; without a label: 0

#### Real neurons only the language model flags

**1b:quarter:real::247**, MLP neuron(s) 1026 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["reflexive pronoun ending a verb phrase (himself, themselves, his after verbs like recused, reminded, rid, acquitted)", "multi-word idiom or correlative completion (\"not only\", \"missed out on\", \"thin on\")", "mid-word subword fragment inside a longer word (var[[ic]]osity, entfer[[n]]ten, sort[[ir]], DEALLOC[[ATE]])", "code or markup token boundary (errors, Kotlin/Go identifiers, XML tag close, LaTeX delimiter)"]}`
- ` trout with a cinnamon coffee rub. Kevin, who had made the final four in the Quickfire, too, acquitted[[ himself]] well enough to be`
- ` Youth Theatre, acro-balancers and free runners from Circomedia and more.<|endoftext|>Previous Offer⏎⏎Sorry[[ you]] missed this offer...`
- ` would start and uproar. One of my guards said they had overheard them talking and discussing how they might rid[[ themselves]] of Him. He`
- ` this regulation and from certain provisions of the Privacy Act:⏎⏎(a) The purpose of the following specified general exemption[[ of]] polygraph records is`

**1b:quarter:real::557**, MLP neuron(s) 2216 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Newline or comment token right after an opening brace of a code block, especially switch statements and function or if bodies", "Mid-phrase tokens in Japanese subtitle or dialogue text, such as particles and verb endings", "Tokens inside LaTeX math expressions, such as macro names and formatting commands", "Function words or adverbs in English narrative prose, such as 'as', 'all', and 'always'"]}`
- `9')⏎            state->level = *mode - '0';⏎        else⏎            switch (*mode) {[[⏎]]            case 'r`
- ` need to cast and allows for possible different states later.⏎    //⏎    switch (ButtonId)⏎    {[[⏎]]        case Hid`
- `. One sort for the ordinals $<\alpha$, the other for the first order situation. Indeed, the system $({\[[bf]] Mod}(\Sigma_{\`
- `{gathered}⏎    {{\mathbb{E}}_{x_{0:t},\tilde x_{0:t}}}[{\[[mathds]]{1}}\!\`

#### Real neurons only the network reader flags

**1b:quarter:real::159**, MLP neuron(s) 642 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token that completes a fixed multi-token expression, such as a word fragment, a proper-noun title, or a code idiom like `new X()`, where the next token is highly predictable from the prefix"]}`
- ` of this article.)⏎⏎One of the best final seasons I’ve ever seen, the last 10 episodes of The[[ Americans]] circled back to`
- `.⏎⏎Read more⏎⏎The big thing missing from Europe’s solution to the refugee crisis? Refugees⏎[[⏎]]A British exit could`
- `itius⏎⏎Category:Australian region cyclone disambiguation pages<|endoftext|>King of Avalon: Dragon Warfare 4[[.]]1.0 (`
- ` document called “Rebuilding America’s Defenses: Strategy, Forces and Resources for a New Century” published by The[[ American]] Enterprise’s “`

**1b:quarter:real::163**, MLP neuron(s) 658 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["Token describing a research study, survey, or poll (e.g., \"study\", \"conducted\", \"survey\", \"poll\", \"analyzed\", \"asked\")", "Token completing a \"last\" recency phrase (e.g., \"last-[minute]\", \"last [year]\", \"last [time]\")"]}`
- `'["ValuesOnGrid"] /. sol // First; (* values of the derivative   x'  *)⏎Split[[[]]Flatten@Position`
- ` the game, why, and how. I did have some introductory text at the beginning, but that was a last-[[minute]] thing. More importantly`
- `⏎so that means select the first and third option in a select-multiple form ?⏎⏎A:⏎⏎Did[[ you]] mean or?⏎`
- ` easy striking distance of both Edinburgh and Glasgow.⏎⏎Perth⏎⏎1Top Sights⏎⏎1Ferg[[uss]]on GalleryC4`

#### 50/50 pools the language model calls one job

**1b:quarter:pooled:50/50:519-984**, MLP neuron(s) 2062, 3946 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["token inside or completing a hyphenated compound modifier, often a genotype or cell descriptor (e.g., -deficient, -expressing, -null, -containing, -infected)"]}`
- `iences in this unmistakably Tudor town range from the touristy (medieval re-creations and Bard-the[[med]] tearooms)`
- ` showed that the activation of AMPK by metformin is abolished in a cell line stably expressing AMPK complexes that contain an AMP-ins[[ensitive]] γ2 mutant;`
- `",⏎      "subota"⏎    ],⏎    "ERANAMES": [⏎      "Pri[[je]] nove ere`
- ` }_3,{\alpha }_2,{\alpha }_1\}$, and appending in each step a sum of a pair of positive[[ roots]] which is greater than`

**1b:quarter:pooled:50/50:122-1958**, MLP neuron(s) 539, 7881 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Subword fragment inside a word or name split across tokens, where the marked token completes the word (e.g., Galaf[[old]], Mu[[eller]], Orm[[ond]], RU[[LE]])"]}`
- `Crim.P. 121(A)(2)(b).⏎J-S55030-18⏎⏎⏎ine[[ffect]]ive assistance. The`
- `⏎The development of these new antimicrobial agents was carried out by a team of eight researchers from the Queen's University Ionic[[ Liquid]] Laboratories (QUILL`
- ` impressive, I would say.⏎⏎Also to note about tonight’s game; Mike Knuble, David Steck[[el]], Matt Bradley,`
- ` SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'DELIMET[[ER]] $$⏎CREATE TR`

### pythia-1.4b, quarter

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 12% | 6% | 10% | 14% | 6% |
| network | 43% | 68% | 64% | 48% | 30% |
| llm | 33% | 56% | 58% | 50% | 22% |

Language-model labels: {'real': {'A': 55, 'B': 12, 'C': 33}, 'pooled 50/50': {'C': 28, 'B': 10, 'A': 12}, 'pooled 70/30': {'B': 6, 'C': 29, 'A': 15}, 'pooled 90/10': {'A': 20, 'C': 25, 'B': 5}, 'related': {'C': 11, 'A': 36, 'B': 3}}; without a label: 0

#### Real neurons only the language model flags

**1.4b:quarter:real::89**, MLP neuron(s) 347 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["the letter 'z' inside the name 'Jezmine'/'Jezebel' in narrative fiction", "opening brace or set-delimiter tokens in LaTeX math notation", "C++ or Objective-C delete statements before a pointer or array name", "the 'select' attribute in XSLT copy-of elements", "PHP or Ruby source code tokens such as variable assignments and string-handling syntax", "prepositions or articles inside English prose, such as 'of', 'onto', and 'one'", "modal verbs like 'should', 'can', and 'is' in English sentences", "fragments of URLs, identifiers, and citation markers such as 'bib' references"]}`
- ` with one's parent and one's child.⏎⏎We would like to extend our appreciation to Michelle Neoh, Je[[z]]ebel Chin Sy`
- `inline>⏎    </xsl:when>⏎    <xsl:otherwise>⏎      <xsl:copy-of[[ select]]="$title"/>⏎`
- ` hands glided over her perfect ass and onto her thighs, where he guided her legs apart as he quickly got on his[[ knees]] behind her. His`
- `⏎U.S. Pat. No. 3,800,938 to Stone discloses elongated frame members including an inclined[[ web]] for close spacing of`

**1.4b:quarter:real::131**, MLP neuron(s) 547 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Word fragments and terms for blood vessels and the cardiovascular system (vascular, cardiovascular, angio-, endothelial cells, aortic/vessel terms)", "The token 'DEBUG' inside #ifdef preprocessor directives in C code"]}`
- `0031C25]\].⏎⏎This study tests the ability of recombinant PP2A subunit from the parasitic nematode *An[[gi]]ostrongylus`
- ` down validates with findings that have been reported previously \[[@R43]\]. Studies have investigated the mechanism whereby OPG could promote[[ angi]]ogenic behavior of endothelial`
- `mode_count_minus_1 = ilog(vorbis_mode_count - 1);⏎#ifdef[[ DEBUG]]_SETUP_`
- `}⏎@@ -646,8 +646,8 @@⏎    else if(!okay) {⏎ #ifdef[[ DEBUG]]⏎       printf("`

#### Real neurons only the network reader flags

**1.4b:quarter:real::64**, MLP neuron(s) 248 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["a function word or noun right before an expected head or referent in expository text, such as 'family' in taxonomic or protein-family phrases, 'its' or 'the' referring back to an earlier entity, and 'son' after a named person"]}`
- ` identified in potyvirus proteins, other than the RdRp, match those encoded by astroviruses; the non[[structural]] protein of a human`
- `rella patens. Control of auxin levels is achieved in higher plants via synthesis of auxin conjugates by members of[[ the]] GH3 family.`
- `:⏎        run_name = mod_name⏎    pkg_name = mod_name.rpartition('.'[[)[]]0]⏎    `
- `d ed. 1970) Judgments, § 79, p. 3240; see, e.g., Lesser &[[ Son]] v. Seym`

**1.4b:quarter:real::77**, MLP neuron(s) 301 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token inside a noun phrase where the head noun, word completion, or 'one'/'a' substitute is being determined"]}`
- ` style="overflow:scroll">⏎The quick brown fox jumps over a lazy dog.                       [OK, a pang[[ram]]]⏎The quick`
- ` [@dujmicICHEP] consists of two optically independent regions contained in one stainless steel vessel. Each region is[[ a]] cylinder with 30 cm`
- ` cases, the reductionist can point to the obvious similarity in electronic configurations between an atom from group _n_ and one[[ from]] group _n_`
- ` from cold to cool.⏎⏎Brighton is without doubt Britain's most colourful and outrageous city, and one[[ with]] many faces. Here`

#### 50/50 pools the language model calls one job

**1.4b:quarter:pooled:50/50:1799-627**, MLP neuron(s) 7235, 2629 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["a token that continues or completes a word, name, or identifier that is split across subword tokens (e.g., 'Shrew[[s]]bury', 'Pend[[le]]', 'Chic[[x]]ulub', 'request_mem_[[region]]')"]}`
- `rho }_j(a)}(\cdot ))$. Since $\s_j^a:R^a_+\setminus \[[{{\]]alpha }_j\}\`
- `word;       /* Internet Explorer 5.5+ */⏎    color: #555;⏎    border:1px[[ solid]] #ccc;⏎`
- `日本でも、すでに大麻由来成分を含んだ製品を手[[に]]入れることが`
- `www.budlitfest.org.uk/ with a pre-festival event on 5 September, when Michael Mor[[pur]]go will be the`

**1.4b:quarter:pooled:50/50:1011-1190**, MLP neuron(s) 4319, 4999 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-word subword fragment that completes a proper noun, especially a British or other place name (e.g., Downs, Taunton, Carmarthen, Corfe)"]}`
- `tland in January 1821 and left on 16 February. Salisbury called at Buenos Aires and arrived in the Down[[s]] on 13 May and`
- `icken employees, coming downtown. But he also is grateful for social media, which has allowed his business to get the word[[ out]], without having`
- `. 391 (Base) Squadron.⏎⏎The flight's regular duty consisted of a courier service to Pus[[an]] and Kimpo in`
- `01752-344840; www.stelizabeths.co.uk; Longbrook St, Plympt[[on]] St Maurice; d`

### pythia-2.8b, quarter

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 14% | 8% | 12% | 14% | 4% |
| network | 46% | 64% | 74% | 50% | 34% |
| llm | 47% | 64% | 64% | 54% | 36% |

Language-model labels: {'real': {'A': 41, 'B': 12, 'C': 47}, 'pooled 50/50': {'C': 32, 'A': 10, 'B': 8}, 'pooled 70/30': {'C': 32, 'A': 9, 'B': 9}, 'pooled 90/10': {'C': 27, 'A': 20, 'B': 3}, 'related': {'C': 18, 'A': 28, 'B': 4}}; without a label: 0

#### Real neurons only the language model flags

**2.8b:quarter:real::136**, MLP neuron(s) 648 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Operators and separators inside algebra and arithmetic problems (exponents, equals signs, line breaks before simplified answers)", "Segments of multi-part gene, protein, and cell-line names and abbreviations (e.g., MDA-MB, IGF, GLP-1, IL-1, MiG-31, Chandrayaan)", "Identifier and section-number fragments in journal article anchor links and citation IDs (e.g., micromachines-10-00, mrm27594-fig, sec2-ijerph-17)", "CSS and HTML display-property tokens (e.g., display: inline-block)"]}`
- ` g = 101 - 63. Suppose 4*v = -5*s - 46, -v - s - g[[ =]] 3*v.`
- ` block the metastatic progression of TNBC cells in mice. To this end, we injected mice subcutaneously with MDA-[[MB]]-231 cells labeled`
- `r*r)**(-36)) assuming r is positive.⏎r**(275/3)⏎Simplify (o*o[[**]]14*o/`
- ` 1763 = s, -5*s = 5*j - 1755. Suppose -5*g - s[[ =]] -90. Let`

**2.8b:quarter:real::353**, MLP neuron(s) 1721 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["subscript and index tokens in LaTeX math expressions", "numeric values and quantities in scientific or statistical reporting", "word-internal or name fragments in English prose and legal text"]}`
- ` with $0\le m_1\le -c^a_{13}$ and $0\le m_[[2]]\le -c`
- `MB-45. CD95L was positive only in melanomas in a more advanced phase than stage I, whereas H[[MB]]-45 was not`
- ` 13), worry (items 2 and 12) and irritability (items 5 and 9), and for items measuring the social[[ inhibition]] facets discomfort in social`
- ` or more), including butter (a food that’s been debated in the Paleo community for a while), and moderate[[ protein]] consumption.⏎⏎`

#### Real neurons only the network reader flags

**2.8b:quarter:real::122**, MLP neuron(s) 563 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token at a boundary of quoted or attributed material, such as closing quotation marks, punctuation just before a closing quote, or the verb or phrase introducing a named speaker (e.g., 'said', 'wrote')", "Token inside citation or reference markers, such as author-year citations, bracketed citation keys, and scripture or source names"]}`
- ` for wanting to make as much money as they can? Caveat emptor, after all.⏎⏎Remember the immortal[[ words]] of P.T`
- ` war rarely conforms to slogans. As General Eisenhower, someone with hard-earned knowledge on this subject,[[ said]] at this ceremony in`
- `, intuition and/or experience are not sufficient in themselves. Communication is not a gift but a skill that can be learned[[ \[[@]]CR30]\] and`
- ` model book assumed in fraternity nationalism is just restarted to a basically more involved dust, the transformation winning of encounter detectors[[.]] 15) focused that`

**2.8b:quarter:real::192**, MLP neuron(s) 921 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["'of' or 'in' right after a numbered section, article, proposition, table, figure, or page reference (e.g., 'Article 11.2 [of]', 'Proposition 7.1 [in]')", "a slash or comma inside a formatted number, code, fraction, or date (e.g., '164[/]', 'MR/S01165X[/]', 'April 3[,] 2008')"]}`
- `le 0$, which implies Lemma \[l:moment\].$\Box$⏎⏎The following inequality, borrowed from page 82[[ of]] Petrov [@`
- `* (\[nlremainder\]) *is satisfied.*⏎⏎Proof: Follows by Proposition 7.3.3[[ of]] Luenberger (`
- ` take an active role in meeting the nation's housing needs.⏎Additionally, the federal Defendants argue that the familial status provision[[ of]] the Fair Housing Act`
- `, phosphate buffer pH 7.5) from extended release tablets containing HPMC ([@CR18]). Use of confocal laser[[ scanning]] microscopy revealed that the`

#### 50/50 pools the language model calls one job

**2.8b:quarter:pooled:50/50:1085-627**, MLP neuron(s) 5277, 2919 (MiniLM True, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["token inside a formulaic, highly predictable template where the next token is nearly determined by convention", "U in U.S.C./U.S. Patent citations, just before the statute or patent continuation", "T separating date and time in ISO-style timestamps", "comma or time separator inside opening-hours listings", "bitwise & in C/C++ bit-field extraction macros"]}`
- `right),$$ where $$\begin{aligned}⏎\Theta_{11}=&72{{\rm Diag}}'({\mathbf}[[P]]) {\left(`
- ` of Neanderthal skull-cap.⏎1857–1881. Henri Milne-Edwards (French,[[ 1800]]–1885)`
- `⏎            </phone>⏎    </contactInfo>⏎    <webLinks lastUpdated="2016-11-07[[T]]05:00:`
- `aaaaaaaaaaaaaaaa\",\"unique_name\": \"unique_name\",\"date_updated\": \"2015-07-30[[T]]20:00:`

**2.8b:quarter:pooled:50/50:1053-625**, MLP neuron(s) 5140, 2914 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Word (verb, adjective, or noun) whose following governed preposition or particle is highly predictable, such as 'proud [of]', 'hang on [to]', 'shy [about]', 'a few [of]'", "Verbs of burial or signing ('buried', 'sign', 'signed', 'signatures') that take a predictable complement"]}`
- ` treatment protocols are needed to adequately estimate associations between genetic polymorphisms and treatment outcomes.⏎⏎Conflict of interest: None⏎⏎[[^]]a^Evangel`
- ` 343/CNBS. It was explained to the participants and their relatives the objectives of the study and they were asked to[[ sign]] the informed consent after`
- ` and pulled out a long, thin object wrapped in a cloth. He handed it to me and said, 'Hang on[[ to]] this. Just in`
- ` usually this "depreciation allowance" is spread over five years or so, whereas this special deal accelerates the full[[ deduction]] to the end of`

## three quarters

### pythia-70m, three quarters

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 16% | 18% | 12% | 22% | 14% |
| network | 49% | 82% | 74% | 44% | 58% |
| llm | 21% | 54% | 56% | 26% | 16% |

Language-model labels: {'real': {'A': 64, 'C': 21, 'B': 15}, 'pooled 50/50': {'B': 7, 'A': 16, 'C': 27}, 'pooled 70/30': {'C': 28, 'A': 17, 'B': 5}, 'pooled 90/10': {'A': 30, 'C': 13, 'B': 7}, 'related': {'A': 36, 'C': 8, 'B': 6}}; without a label: 0

#### Real neurons only the language model flags

**70m:three_quarters:real::664**, MLP neuron(s) 861 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["'Bus' starting a bus route or transport service description (e.g., 'Bus 1/1A', 'Bus 843')", "commerce and market terms ('market' stalls and days, 'business' professionals, occupation)", "road and travel-infrastructure words ('road' safety, 'road' bridges, 'road' length)", "medical and research-study terms ('hospital' staff and patients, 'study' subjects and design)", "pronouns and emphasis words in dialogue or reflective prose ('it', 'you', 'too', 'I', 'we')"]}`
- ` was not purchased for your use only, then you should purchase your own copy. Thank you for respecting the hard work of[[ this]] author.⏎⏎`
- ` by the bishop. I wouldn’t have wanted to invite the whole ward, but that’s just the way I[[ am]]. I only needed`
- ` be more useful than general-purpose ones.⏎⏎“I think an important point to make in terms of how our[[ space]] is evolving too -`
- `. Berneray, North Uist, Benbecula, South Uist and Eriskay are all linked by[[ road]] bridges and causeways`

**70m:three_quarters:real::677**, MLP neuron(s) 881 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Volume number in a legal case citation, right before the reporter abbreviation (e.g., F.2d, U.S., S.W.)", "Newline or whitespace token after an opening XML/HTML tag in code", "Empty table-cell or placeholder token in a formatted data table", "Word following 'I' or quantifier in narrative prose (e.g., 'I had ever met', 'many times')", "Numeric or symbol token inside a statistical or LaTeX expression"]}`
- ` S.Ct. 818, 19 L.Ed.2d 871 (1968); Evans v. Cunningham,[[ 335]] F.2d`
- ` The evidence overwhelmingly establishes the guilt of appellant in this respect. In State *960 v. Prichett, 6[[21]] S.W.`
- ` mode of HIV transmission.⏎⏎###### ⏎⏎Demographics and clinical characteristics of patients with reportable tropism results⏎⏎[[                        ]]                               India           `
- `⏎"I don't want to be disrespectful here, but Tyson Fury is the most unentertaining fighter[[ I]]'ve ever seen,"`

#### Real neurons only the network reader flags

**70m:three_quarters:real::45**, MLP neuron(s) 60 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["partial or split word fragments, where the next subword token completes a word (e.g., 'NE' in 'NE708', 'Temp' in 'Tinie Tempah', 'cruc' in 'crucifixion')", "the word 'hot' and its derivatives (hot, hotly, hotdogs, hot air), used in varied senses", "function words such as 'to', 'be', 'at', and 'in' inside fixed or idiomatic phrases (something to do, not to be confused, at all, right in the first place)"]}`
- `⏎qui non qui siamo solo seduti⏎fare una pausa e nessun altro lo era⏎[[in]] giro e ho`
- ` suo padre non il⏎Nick è in gravi difficoltà oggi non otterrà⏎[[in]] tutto ciò`
- ` здесь Элвис⏎⏎Italian: ⏎vuoi che tutti disturbino tutto⏎[[qu]]indi è quello`
- ` woman of the 21st century now, and I want her stories to reflect the world in which we live, rather than[[ ret]]elling stories from her`

**70m:three_quarters:real::98**, MLP neuron(s) 131 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["word-piece fragments that begin a word or compound whose continuation is predictable, such as 'mid' in time-period phrases (mid-October, mid-1960s) and 'Sur' in words like Survivor and Surprise", "'fall'/'fell' verb forms in quantity or outcome phrases (falls far short, fell far short)", "superscript caret marking ion charge in chemical notation (Ca^2+^)", "replacement or garbled character inside mis-decoded non-English text"]}`
- `bisphosphate (PIP2). It was hypothesized \[[@B44-brainsci-03-01374]\] that abnormal Ca[[^]]2+^ entry and`
- ` the specific proteolytic activity of MMP-1 ([@b14-ehp-119-784]).⏎⏎Ca[[^]]2+^ imaging of`
- ` Jackson's 'Petticoat Affair'", The History Net, June 1999⏎This American Life, #485 "[[Sur]]rogates", Act`
- `iphenyl ether and the benzene.⏎Where xe2x80x94Axe2x80x94B[[xe]]2x80x`

#### 50/50 pools the language model calls one job

**70m:three_quarters:pooled:50/50:420-992**, MLP neuron(s) 549, 1299 (MiniLM True, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token inside a formal technical or scientific phrase where the next word is highly predictable from a fixed collocation, such as a passive auxiliary before its participle (\"is [carried] out\", \"was performed\") or a determiner before a set phrase (\"in [the] past decades\", \"and [the] like\")"]}`
- `. [@koch2007], the Hamiltonian of the transmon is analogous to that of an electrically charged particle whose motion[[ is]] restricted to a circular`
- ` input detecting means; and motor driving means for supplying current to the brushless motor, wherein the calculation of the angular position[[ is]] carried out with reference`
- `int. Finn Juncker uddannede gambianeren i hans butik og havde medbragt en[[ hel]] del kostbare`
- `kyttet med indførelsen af revisionsproceduren, der sikrer en retfærd[[ig]], ikkedisk`

**70m:three_quarters:pooled:50/50:242-1213**, MLP neuron(s) 317, 1613 (MiniLM True, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["the final fragment of a word or number split across tokens, where the model completes the rest of a partially written word or multi-digit number (e.g., page numbers in legal pin cites, word endings in French, Spanish, and Portuguese)"]}`
- ` chaque jour trois fois, pleurant et priant sous le bel arbre, et toujours un petit o[[ise]]au blanc ven`
- `, si bien han tenido fuertes pérdidas a causa de condiciones climáticas advers[[as]], como en el`
- ` immediate appeal is the only effective way to challenge it.  Id. at 84, 101 S.Ct. at 9[[96]]-97.  `
- ` on the "terms, conditions, or benefits" of employment. Von Gunten, 243 F.3d at 8[[65]].⏎Alber`

### pythia-160m, three quarters

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 13% | 6% | 24% | 12% | 18% |
| network | 42% | 72% | 66% | 56% | 50% |
| llm | 18% | 50% | 56% | 34% | 18% |

Language-model labels: {'real': {'A': 70, 'B': 12, 'C': 18}, 'pooled 50/50': {'A': 14, 'C': 25, 'B': 11}, 'pooled 70/30': {'A': 16, 'C': 28, 'B': 6}, 'pooled 90/10': {'A': 25, 'B': 8, 'C': 17}, 'related': {'A': 35, 'B': 6, 'C': 9}}; without a label: 0

#### Real neurons only the language model flags

**160m:three_quarters:real::467**, MLP neuron(s) 688 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Prepositions or particles completing verb/adjective collocations (e.g., 'used to', 'familiar with', 'pointed out')", "Mid-word subword fragments that complete a split word (e.g., 'Har|vest', 'ext|inguish', 'Megam|all')", "Code syntax tokens such as closing brackets, whitespace indentation, and operators", "Tokens completing numbers, identifiers, or specialized terms in technical text (e.g., '300 MHz Infinity+', 'UV--Vis', 'Local 2850')"]}`
- `⏎      typeof object.nodeType === 'number' &&⏎      typeof object.nodeName === 'string'⏎[[  ]]));⏎}⏎`
- ` (MFQ) has been suggested for combination with ART and PPQ given the inverse effects PPQ and MFQ pressure[[ have]] on *pfm`
- ` jobs back.⏎⏎"For the workers it feels like a relief to be believed in some way," said Local 28[[50]] organizer Sarah Norr`
- ` on smartphones, at 84.6%. As a result, the overall Index abandonment rate rose 1.4% year over[[ year]] to 76.0`

**160m:three_quarters:real::1350**, MLP neuron(s) 2027 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["'been' (or 'has'/'have') in present-perfect constructions predicting a past participle", "colon or number tokens in antibody dilution and ratio notation (e.g., 1:1000)", "mid-word subword fragments inside split words (e.g., 'Mag[[ento]]', 'Xam[[arin]]')"]}`
- `...<|endoftext|>Comment:⏎A brand-new, unused, unopened video game in perfect condition in its original packaging and[[ with]] all original packaging materials`
- ` would be a daunting task, especially if you're not familiar with microcontroller basics.⏎This site is not suited for[[ product]] recommendations, but I`
- ` work with each other. They are of a piece.⏎⏎The Simplenote web app is hosted on Google[[ App]] Engine. In my`
- ` mammalian cells. Since inducer molecules are not expected to have targets in mammalian cells, the possibility of interference with cellular processes[[ is]] reduced.⏎Of`

#### Real neurons only the network reader flags

**160m:three_quarters:real::62**, MLP neuron(s) 94 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Pre-head modifier or determiner inside a noun phrase, especially comparative 'more'/'most' and possessives like 'its', 'their', 'his', 'her', where the next word is an adjective or noun being modified"]}`
- `⏎Traditionally, in open source, there was a lot of emphasis on singular projects. Today, it's much[[ more]] about how multiple communities`
- ` equivalent) and physically counting the shoppers present when the messages air.” Peter Kosann states, “POP Radio has a[[ huge]] audience, unique platform`
- ` which are many on the web which exist without problem. Training On Line These on-line tutoring careers have gotten a[[ greater]] desire within the nowadays`
- `ick, Ayn Rand (though she was in fact an Objectivist and would loathe being called a libertarian, but[[ her]] political philosophy is essentially`

**160m:three_quarters:real::206**, MLP neuron(s) 306 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["The `%` character inside printf-style format strings in C and similar code, where a conversion specifier is expected next", "The `#` token starting a preprocessor directive, header, or file/section boundary in source code or a document after a break"]}`
- `0, uint32_t a1, uint32_t a2)⏎{⏎	qemu_log[[("%]]s: a0`
- `99, &c);⏎    hollow_list_write(&h, 1, &d);⏎⏎    printf[[("%]]d\n",`
- ` there is limited content available. Additionally, true 3D content requires multiple cameras to film objects from different viewpoints.<|endoftext|>[[#]] /* **************************************************************************`
- ` -triple i386-apple-macosx10.7.2 < %s | FileCheck %s[[⏎]]⏎// The preferred`

#### 50/50 pools the language model calls one job

**160m:three_quarters:pooled:50/50:1947-1365**, MLP neuron(s) 2925, 2051 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["the connector token (to, -, --, –, and) between the start and end of a numeric, date, or time range, where the next token is predicted to be the upper bound"]}`
- `West, Geelong, Melbourne's north-west and Hume are some of the most difficult areas to find work for 15[[ to]] 24-year-`
- ` (CA-95-318-3-P)⏎⏎Submitted:   May 17, 1997                   Decided:   [[May]] 28, 1997⏎⏎`
- `code:GetPlugin3X86Dir}"; Components: main_far3_x86; Flags:[[ ignore]]version⏎Source`
- `(new ModuleCategoryViewControl("Item6", "A", typeof(int)));⏎        items.Add(new[[ Module]]CategoryViewControl("`

**160m:three_quarters:pooled:50/50:452-1945**, MLP neuron(s) 666, 2923 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token where the next word is highly predictable from a fixed phrase or collocation (e.g., \"in force\", \"into compliance\", \"in error\", \"virtual file\" → \"system\", \"National Hockey\" → \"League\", \"Stage-co\" → \"ach\")"]}`
- ` or why this should be done at this time, long after the case was settled and dismissed with prejudice in its entirety is[[ not]] explained. Even if`
- ` he had. Nor did the petitioners who sought relief from the state supreme court in original habeas cases, for it has long[[ been]] the law that a`
- `, and beads in her hair. Moonglow was in full gothic mode, and it must have taken some[[ time]] to achieve the effect`
- ` in a sling, but held a revolver in his⏎right hand.⏎⏎That silent cabin evidently puzzled them[[,]] and they were not`

### pythia-410m, three quarters

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 15% | 22% | 24% | 18% | 6% |
| network | 44% | 48% | 50% | 36% | 30% |
| llm | 40% | 62% | 72% | 42% | 24% |

Language-model labels: {'real': {'A': 51, 'C': 40, 'B': 9}, 'pooled 50/50': {'A': 12, 'C': 31, 'B': 7}, 'pooled 70/30': {'A': 9, 'C': 36, 'B': 5}, 'pooled 90/10': {'C': 21, 'A': 26, 'B': 3}, 'related': {'C': 12, 'A': 34, 'B': 4}}; without a label: 0

#### Real neurons only the language model flags

**410m:three_quarters:real::22**, MLP neuron(s) 39 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Verb or determiner inside 'make/have/show an effect, difference, impact, or sense' idioms, often negated (e.g., 'doesn't make any difference', 'makes no sense')", "Numeric fragments in legal citations and docket numbers (reporter volumes, case numbers, statute sections)"]}`
- `⏎⏎Voice: Are you going to attack me? Well... if that's what you want. It won't do[[ you]] or I much good`
- `u/0/#label/TechDiscuss/141a5c074641dd70Please have a look. Thanks[[ a]] lot.2013/`
- ` or an assembly reference?). I am not sure if I need to add a dll reference or something else. Please could[[ someone]] advise how to solve`
- `am-5pm Mar-Oct;  c) theme park on the headland in the 1980s hasn't done[[ much]] to enhance the view`

**410m:three_quarters:real::113**, MLP neuron(s) 231 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Tokens right before a noun for a document, form, or paperwork item (e.g., 'filling out [forms]', 'print the [Registration Form]', 'executed a [deed of trust]', 'send an [email]')", "Tokens in or before a list or noun phrase naming clothing or apparel (e.g., 'jackets[,] vests', 'a suit [and] 6 shirts', 'wearing [yellow vests]', 'navy top [with]')", "Tokens inside source code, predicting syntax such as operators, brackets, or identifiers (e.g., 'self.titleText.text [!=]', 'View [style=]', 'textarea id[=\"]')"]}`
- ` classes or other activities that are part of planned convention agenda are required to pay the full registration fee.⏎⏎Please print[[ the]] Registration Form above and`
- `(s) and/or boutonniere(s) for the wedding couple⏎• 2 corsages and[[ 2]] boutonnieres`
- `’s birth⏎certificate, his mother’s death certificate, his father’s⏎naturalization certificate, and[[ a]] “memo of`
- ` with his name. A. Lett. took such care marking this suitcase, blocking out the black ground for the carefully hand[[ written]] white letters.⏎`

#### Real neurons only the network reader flags

**410m:three_quarters:real::18**, MLP neuron(s) 30 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Tokens where 'lost', 'lose', 'loss of', or 'losing' (something valued, such as a limb, sensation, memory, control, or life) is expected or occurs", "Tokens in contexts describing damage, impairment, or decrease to a body part or function (e.g., 'damage', 'broken', 'decreased', 'shorter')"]}`
- ` BodyPump for quite some time. He whipped out a picture of himself, a before and after pic. Robert had[[ lost]] over 50 pounds within`
- `. Joint reconstruction or replacement may be required to relieve the resulting pain and restore function. Most joints in the body, including[[ the]] neck, shoulders,`
- ` August 1952 a huge wave of water swept through Lynmouth after torrential rain. The devastation was immense – 34 people[[ lost]] their lives; four`
- ` which is the main reason of treatment failure 3 weeks after membrane placement and exposes the membrane which leads to bacterial infection and decrease[[ in]] the levels of gained`

**410m:three_quarters:real::211**, MLP neuron(s) 441 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["Clause subjects (names or pronouns such as I, he, she, or a character's name) that are followed by a verb of character action, especially after 'when' or 'if' in narrative or technical sentences", "Words right before a predicted gesture or facial-expression word, such as 'smile', 'nodded', 'grin', 'hug', or 'look', as in 'gave her a [smile]'"]}`
- ` working with it, but here is the code there anyway.⏎When I run it, it runs very slow and when[[ I]] scroll by it,`
- `.⏎⏎"How did she manage to get hold of the shoes?" Malveria moaned.⏎⏎Th[[rix]] glanced at her own`
- ` that. i installed it. ⏎i have gone to preferences => edit => set KDE konsole⏎amazingly when[[ i]] open the console of`
- ` appreciated more than just his hard work that day. He was sure of it! 'Anytime you come back,[[ just]] ask for me and`

#### 50/50 pools the language model calls one job

**410m:three_quarters:pooled:50/50:36-140**, MLP neuron(s) 68, 303 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["a function word (mostly 'of', 'the', 'a', 'in', 'to') or a word fragment in the middle of a fixed multi-word phrase, where the next token is highly predictable, such as 'At [the] time', 'out [of]', 'Some [of]'"]}`
- ` that working with the public was the aspect of her job that she disliked the most. Working with the public was at[[ the]] core of the Education`
- ` introduced rules that obliged listed companies to switch their auditor (‘mandatory rotation’) every 20 years.[12] At[[ the]] conclusion of the previous`
- ` cell 160 includes an active front end 162 that serves as a three-phase bridge as it receives power from dedicated three-[[phase]] secondary windings of`
- ` on the ATmega328 (datasheet). It has 14 digital input/output pins (of which 6 can be used[[ as]] PWM outputs),`

**410m:three_quarters:pooled:50/50:1920-1959**, MLP neuron(s) 3854, 3934 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["a token inside or just before a multi-token scientific, chemical, or proper name (e.g., Toxoplasma, actinomycin, cyclosporin, FK506, LRRK2, Milli-Q) where the rest of the term is highly predictable", "a separator or attribution token in formulaic text (e.g., the slash in photo credits like 'GETTY IMAGES' and in URLs or file paths, or 'by'/'in' before a named author, publication, or source)"]}`
- ` (H) Day 7-infected mice, cultured in the presence of PMA, ionomycin, and Brefeld[[in]]-A, and`
- `// visit http://go.microsoft.com/?LinkId=9394801⏎⏎public class MvcApplication[[ :]] System.Web.`
- `'s 19th-century country houses.⏎⏎Balmoral CastleVISITBRITAIN / BRITAIN ON[[ V]]IEW / GETTY`
- ` museums, bars and entertainment venues. Newcastle's nightlife is legendary and revelling in an evening on the tiles here is[[ a]] quintessential experience.`

### pythia-1b, three quarters

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 19% | 26% | 14% | 12% | 14% |
| network | 45% | 76% | 68% | 50% | 50% |
| llm | 34% | 74% | 50% | 54% | 32% |

Language-model labels: {'real': {'A': 59, 'C': 34, 'B': 7}, 'pooled 50/50': {'C': 37, 'A': 12, 'B': 1}, 'pooled 70/30': {'B': 5, 'C': 25, 'A': 20}, 'pooled 90/10': {'A': 15, 'C': 27, 'B': 8}, 'related': {'C': 16, 'A': 31, 'B': 3}}; without a label: 0

#### Real neurons only the language model flags

**1b:three_quarters:real::191**, MLP neuron(s) 789 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Legal and court text, such as judge and justice titles, 'Entered for the Court', and opinion boilerplate", "Hexadecimal hashes and numeric tokens inside data dumps, logs, and alignment outputs", "Source code tokens, such as method names, parameters, and identifiers in C#, Java, JavaScript, and Objective-C", "Spanish-language prose, where the token is a fragment 'l' before a verb stem such as 'levó' or 'lleva'"]}`
- `, Michigan⏎⏎  September 12, 2017                                                                                   Stephen J. Markman,⏎                                                                                                                 [[Chief]] Justice⏎⏎                                                `
- ` carry out its statutory duties.⏎⏎⏎12⏎ GTE refers us to Justice Brandeis's dissent (joined[[ by]] Justice Holmes) in`
- ` Armore, Washington, D. C., of counsel), for Commissioner of Patents.⏎Before RICH, Acting[[ Chief]], MARTIN`
- ` current cretins in power could conjure up: They were removed and arrested by police, Chairman Orrin Hatch[[ told]] them to "shut`

**1b:three_quarters:real::1033**, MLP neuron(s) 4226 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["member access punctuation (dot or arrow) right before a property or method name in code, especially exception, response, and error objects", "mid-word subword fragments in prose, such as partial words like 'ort', 'sp', 'ado', and 'amar' that are completed by the next token", "numeric and unit contexts in scientific or technical text, such as concentrations, counts, and thresholds (e.g., 'mM', '10', '4 minutes')"]}`
- `::cerr << "Error trying to load config file " << filenames[k] << ": " << e[[.]]what() << std`
- `Membership/UserRegistration.aspx");⏎}⏎catch(Exception ex)⏎{⏎   string s = ex[[.]]ToString();⏎   `
- ` both user and weather as an argument.⏎⏎function getWeather ( user ) { return new Promise ( ( resolve[[ ,]] reject ) => {`
- `) se movilizará mañana en el acceso del puente Rosario-Victoria para recl[[amar]] medidas de ap`

#### Real neurons only the network reader flags

**1b:three_quarters:real::209**, MLP neuron(s) 835 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-phrase tokens where the next word or word-piece is highly predictable, such as word fragments being completed (Hiro[shi], sildenafil[s], rec[tion], ad[enoid]) and syntactic connectors (relative 'that', infinitive 'to') that hand off to a continuation"]}`
- `). Results demonstrate that this formulation induces a strong protective immunity in the infected and vaccinated animals. The rapid expulsion of the[[ intestinal]] parasites in the PP`
- ` and heat loss, spa owners have long utilized covers for their spas. The spa covers are generally sized and configured to[[ completely]] cover the spa.`
- ` keep everything in RAM in the first place) because one of the greatest protections of the Java based Android is the sandbox[[ that]] the OS creates for`
- `ans have been one of the corner stones of the Java EE specification. As a Java EE 5 certified application server, Apache[[ G]]eronimo supports E`

**1b:three_quarters:real::382**, MLP neuron(s) 1456 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Coordinating conjunction (and/or/y/e/ni/but) or the position right before the second item of a coordinated or parallel pair or list, e.g. 'insight [and] strategy', 'injury [or] damage', 'conductas [y] comportamientos'"]}`
- ` gone to war,⏎and to protect it we would go to war again . . .⏎⏎Only second to liberty[[ is]] the maintenance of peace`
- `; on another level, the economy of the nation, the empire of business within the republic, both include in their basic[[ premise]] the concept of perpetual`
- `es, ni expulsiones).⏎⏎La realidad es que nadie ha pagado hasta ahora por conductas[[ y]] comportamientos mar`
- ` a photograph to the victim the day he was released from the hospital.[10] The victim had not theretofore given the[[ police]] a description of any`

#### 50/50 pools the language model calls one job

**1b:three_quarters:pooled:50/50:1591-660**, MLP neuron(s) 6456, 2686 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Fires on the token that begins or forms a word starting with the letter or sound 's' (including 'sh' and capital 'S'), often right before a word-initial 's' continuation"]}`
- ` analyzed by means of one-way analysis of variance (ANOVA) with LSD test; Otherwise, Kruskal-Wall[[is]] test with a non`
- `jork and S. Pitnick. Intensity of sexual selection along the anisogamy–isog[[amy]] continuum. , 441`
- ` 46) was one of the first primary schools in Nørrebro. It opened in 1875 as a branch of[[ S]]ankt Hansg`
- ` servant confronts Jehovah’s people (v 1) is their desecration of the Sabbath. Even[[ S]]abbath meetings aren’`

**1b:three_quarters:pooled:50/50:1844-319**, MLP neuron(s) 7410, 1224 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-clause tokens, mostly auxiliaries, modals, and copulas (would, may, will, be, was, were, is), where the main predicate or verb phrase is still to come after a subject"]}`
- ` full compliance by Ranier with the plans and specifications called for within the contract, and as vouched for by the architect[[,]] was the final payment`
- ` that screeners who were rude to co-workers and passengers or just generally incompetent but had made it through their probationary[[ period]] could hang onto their`
- ` child participants; (5) all participants must not suffer from any cognitive impairments, hearing or visual impairments or major diseases that[[ would]] prevent them from understanding`
- ` the previously dry valley from 1 year to another can block the most direct route to the village and its clinic and make travel[[ there]] exceedingly difficult. In`

### pythia-1.4b, three quarters

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 27% | 18% | 18% | 26% | 18% |
| network | 46% | 62% | 46% | 56% | 38% |
| llm | 47% | 68% | 60% | 64% | 36% |

Language-model labels: {'real': {'A': 45, 'B': 8, 'C': 47}, 'pooled 50/50': {'C': 34, 'A': 12, 'B': 4}, 'pooled 70/30': {'C': 30, 'B': 7, 'A': 13}, 'pooled 90/10': {'C': 32, 'A': 14, 'B': 4}, 'related': {'A': 24, 'B': 8, 'C': 18}}; without a label: 0

#### Real neurons only the language model flags

**1.4b:three_quarters:real::321**, MLP neuron(s) 1376 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["subword fragments that continue a split word (e.g., 'Re' in Realtors/Replacement, 'age' in bandage, 'post' in postjunctional)", "auxiliary or copular verbs right after a subject noun phrase (e.g., 'Activated' 'was', 'Errors were', 'Mobile Board of Realtors')", "colons or punctuation that introduce an error message or explanation (e.g., 'Graph disconnected:', 'ERROR: Job failed:')", "adverb or manner-word endings in narrative prose (e.g., 'lazily', 'firstly', 'buckled')"]}`
- `, which Fox released years ago.<|endoftext|>Wunnamurra Shire⏎⏎Wunnamurra Shire[[ was]] a local government area`
- ` slowly. "I don't know about that. I've never been renowned for my bravery."⏎⏎Jamie[[ hooked]] his thumbs in his`
- `ard, the⏎other three scattered and took to cover along the edge of the clearing.⏎Ballard himself, rifle[[ under]] his arm, stepped`
- ` or openings, that are specifically designed for certain surgical procedures are well known. Typically, such drapes are structured with pre[[-]]shaped and pre-`

**1.4b:three_quarters:real::381**, MLP neuron(s) 1609 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["'do' in 'Why would you/he do that?' dialogue", "mid-word fragments in code constants and identifiers (e.g., PT_READ_D, IP_DUMMYNET_)", "closing LaTeX math delimiters ($ and similar) after math expressions", "'was'/'like' in quoted reported speech ('I was like')", "figure/citation reference tokens in scientific text (e.g., Fig. 5C, ref)"]}`
- ` the Relationship?⏎⏎Some might say that if the sex disappears, the relationship is already dead. I wouldn’t[[ go]] that far, though`
- ` really bad automobile accident..." "But you're gonna be okay." "You're gonna be fine." You know, stuff[[ like]] that." "She`
- ` (OVA), cytochrome *c*, Tamm Horsfall glycoprotein and HIV-1 gp120, have been attached to[[ pal]]mitic acid via`
- ` and small.⏎⏎I doubt we'll see magazine spoilers for the other girls' routes, and frankly I prefer[[ it]] that way. Too`

#### Real neurons only the network reader flags

**1.4b:three_quarters:real::200**, MLP neuron(s) 859 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["position inside a multi-token word, name, or technical term (often medical) where the next token completes it"]}`
- ` do with him now. Few students tried to hold a gathering today over his death, 3 students suspended,” AMU regist[[rar]] Abdul Hamid told`
- `, which is at the basis of the onset of papillary carcinoma.⏎⏎Our choice to perform a radical surgery instead[[ of]] the subtotal thy`
- ` But she has some thoughts on the problems in the budget.⏎⏎Benedict, who was on chemotherapy for bladder[[ and]] prostate cancer at the`
- `ating disputes within the community which served as an avenue to discuss and address social problems affecting Indian workers. It set up a[[ pan]]chayat committee`

**1.4b:three_quarters:real::213**, MLP neuron(s) 922 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Positions in the 'it' anticipatory/extraposed construction, such as the comma after a fronted phrase or the 'it' itself (e.g., 'In light of these figures, it is doubtful'), and idiomatic 'it' phrases (make it, battle it out, cut it, when it comes to)"]}`
- `⏎⏎We offer you with unique and best of start-up strategies⏎⏎Each of our team members is highly knowledgeable[[ when]] it comes to rendering`
- `ON / GETTY IMAGES ©⏎⏎### Cornwall's Coast⏎⏎In Britain, you're spoilt for choice[[ when]] it comes to beautiful`
- ` man crossed the water the dwarf would have torn him to pieces. As it was he died within the year, in consequence[[,]] it was supposed,`
- ` vehicle such as a jeep or tractor which does not have the appropriate power source necessary to raise the elevator. In these[[ circumstances]], it is very`

#### 50/50 pools the language model calls one job

**1.4b:three_quarters:pooled:50/50:650-602**, MLP neuron(s) 2687, 2492 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token that is a partial fragment of a longer word, name, or abbreviation, where the word continues across the token boundary"]}`
- `(a).⏎When the criminal appeal rules govern, Rule 304(a) does not apply. In re D.[[D]]., 337 Ill.`
- `, Inc. v. FCC, 104 U.S.App.D.C. 226, 261 F.2[[d]] 55, 65 (`
- ` Brooke McDonald to slide one past the keeper. Rhiannon Walding scored next as she lofted a well[[ weighted]] ball from near half`
- ` the staff and visitors at Keith Tulloch Wine in Pokolbin in the Hunter Valley yesterday, who were caught un[[aw]]ares when a tornado`

**1.4b:three_quarters:pooled:50/50:1204-190**, MLP neuron(s) 5009, 808 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["a subword fragment or character that is the middle or end of a word, identifier, hex literal, or code token split across tokens (e.g., 'fluorescent[[ly]]', 'elect[[rol]]ess', 'R[[itch]]ie', '0xc[[3]]', 'FAM/HE[[X]]')"]}`
- ` a superseding cause of harm to⏎        another which the actor’s negligent conduct is a⏎        substantial factor in[[ bringing]] about, if⏎⏎`
- `:basic-link>⏎    </xsl:when>⏎    <xsl:otherwise>⏎      <xsl:[[call]]-template name="`
- ` weekly pass is £20.20. Children ride for free. At stops with yellow signs, you must buy your ticket[[ from]] the automatic machine (`
- ` routers, switches, gateways, and the like.⏎With this arrangement, when the supporting cell site equipment receives[[ from]] the network infrastructure data`

### pythia-2.8b, three quarters

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 19% | 20% | 16% | 18% | 10% |
| network | 61% | 42% | 52% | 46% | 42% |
| llm | 43% | 50% | 56% | 62% | 36% |

Language-model labels: {'real': {'C': 43, 'A': 39, 'B': 17, None: 1}, 'pooled 50/50': {'A': 18, 'B': 7, 'C': 25}, 'pooled 70/30': {'C': 28, 'A': 20, 'B': 2}, 'pooled 90/10': {'C': 31, 'A': 17, 'B': 2}, 'related': {'A': 26, 'B': 6, 'C': 18}}; without a label: 1

#### Real neurons only the language model flags

**2.8b:three_quarters:real::384**, MLP neuron(s) 1935 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["UK travel guide text, marking place names and transport/route details", "arithmetic and algebra problems, marking numbers or operators", "open-source software license boilerplate", "US court and legal filing text", "news reporting on sports, politics, and events"]}`
- `                 Defendants.⏎⏎⏎⏎Appeal from the United States District Court for the District of⏎Maryland, at Greenbelt.[[     ]]Peter J. Mess`
- ` Windsor & Eton⏎⏎#### Bray⏎⏎#### Henley-on-Thames⏎⏎### Bath &[[ Southwest]] England⏎⏎####`
- ` quieter times, and more expensive if you buy your ticket on the spot and it's Friday afternoon. As a guide[[,]] a 200-mile`
- `⏎ *⏎ * This library is free software; you can redistribute it and/or⏎ * modify it under the terms[[ of]] the GNU Lesser General`

**2.8b:three_quarters:real::448**, MLP neuron(s) 2294 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["numeric tokens and operators in arithmetic and sequence-continuation problems", "punctuation and number fragments inside legal, journal, and statistical citations and reported results", "word choice in narrative prose and news-style text"]}`
- ` term of 3896, 7597, 11296, 14993?⏎-q**2 + 3704*q[[ +]] 193⏎What is`
- `39, -652841, -979245, -1305651?⏎-l**2 - 3[[263]]99*l -`
- `The only relevant authority cited in connection with this contention is Hyde Corp. v. Huffines, 158 Tex. 566[[,]] 314 S.W`
- ` a multiple of 16?⏎False⏎Suppose 4*b + 3*k = 8019, -392*b[[ +]] 3*k -`

#### Real neurons only the network reader flags

**2.8b:three_quarters:real::122**, MLP neuron(s) 618 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token completing a fixed multi-word term, name, or formulaic phrase in technical and academic text (e.g., 'Kluwer Academic [Publishers]', 'coronary [artery]', 'customer [relationship] management', 'root [coverage]')"]}`
- ` were selected for this study. They were matched by rank and assigned to either a treatment or control group. The treatment group[[ received]] the intervention three consecutive`
- ` were washed five times with phosphate-buffered saline (PBS) containing 0.05% Tween 80 then blocked with 5[[%]] skimmed milk powder`
- ` institutions in the adolescent myopia prevention and control research dataset based on three measures of centrality: degree, betweenness,[[ and]] closeness centrality`
- ` be more at ease in your caregiving role.<|endoftext|>Clinical significance of intestinal type fatty acid binding protein in patients undergoing coronary[[ artery]] bypass surgery.⏎`

**2.8b:three_quarters:real::136**, MLP neuron(s) 660 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["word or name fragments mid-token that complete a known term, name, or identifier (e.g., 'ira' in Thalassiosira, 'ac' in iliac, 'od' in Codish, 'Ted' in Ted Hoff)", "body-part nouns after a possessive or determiner (e.g., 'mouth' in 'keep your mouth shut', 'lips' in 'his lips tight clenched', 'nose' in 'the nose swollen')"]}`
- `:sbdm\]) is, in the terminology of [@DBLP:conf/ijcai/C[[od]]ishMPS13`
- `aken. Every map is four colourable. , 82:711–712, 1976.⏎⏎M. C[[od]]ish, A.`
- `⏎      }⏎    }⏎  },⏎  "keg_only": false,⏎  "bott[[le]]_disabled": false`
- ` protected right. In rejecting that contention, the Court wrote:⏎But such a reading would make of the Fourteenth Amendment a[[ font]] of tort law to`

#### 50/50 pools the language model calls one job

**2.8b:three_quarters:pooled:50/50:1734-369**, MLP neuron(s) 8706, 1868 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token in mid-phrase (often a preposition, function word, or word fragment) where the next word is strongly fixed by the local collocation or name, such as 'designed to be used as a spectrometer [for]' or 'Caliste-SO... Church [Street]'"]}`
- `A two-day celebration of Suffolk produce and top-class cooking.⏎⏎4Sleeping & Eating⏎⏎ o[[O]]cean HouseB&`
- `ae of other stars.⏎⏎Caliste-SO is a CdTe hybrid detector designed to be used as a spectrometer[[ for]] a hard X-`
- `-malarial drugs, and ART resistance in *Plasmodium falciparum* is one of the greatest threats to global efforts[[ to]] control, eliminate and`
- `⏎⏎Financing for low-carbon and climate resilient development. The draft outcome document says: “Public and private investments[[ in]] innovations and clean technologies`

**2.8b:three_quarters:pooled:50/50:38-637**, MLP neuron(s) 199, 3209 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token ending the first element of a coordinated or parallel pair or list (e.g., 'males' before 'and females', 'with' before 'and without', 'three' before 'and four'), where the model anticipates a parallel counterpart such as 'and', 'or', or a matching second item"]}`
- ` quantum entropies, respectively. Taking the limit of *N* → ∞ copies for Steps *b[[*]] and *c*`
- ` in the WD fed mouse and demonstrated excessive activation of S6K1 as well as increased abnormal Ser phosphorylation of IRS-[[1]] and IRS-2`
- `b+\bar{b}}^{\eta}$ differ significantly in size and shape from their counterparts for pions, kaons,[[ and]] protons as can be`
- `ximal Femur              7.1                                              5.5    1.6⏎  3         Osteosarcoma[[        ]]F         21           `
