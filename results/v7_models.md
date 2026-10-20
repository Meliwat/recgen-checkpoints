# v7 results per model

Flag rates (unrelated mixed) by reader and unit type at the middle layer, then, per model, up to 2 real neurons each reader alone flags and 2 50/50 pools the language model missed, with its stated patterns and four of the unit's 50 snippets (its 1st, 2nd, 26th and 27th strongest; the language model saw all 50 in random order). Unit ids use v6's sample indices; the MLP neuron ids are given beside them.

## pythia-70m

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 17% | 16% | 20% | 10% | 6% |
| network | 42% | 82% | 74% | 62% | 44% |
| llm | 38% | 56% | 52% | 42% | 22% |

Language-model labels: {'real': {'A': 53, 'C': 38, 'B': 9}, 'pooled 50/50': {'C': 28, 'A': 13, 'B': 9}, 'pooled 70/30': {'A': 18, 'B': 6, 'C': 26}, 'pooled 90/10': {'C': 21, 'B': 6, 'A': 23}, 'related': {'A': 29, 'B': 10, 'C': 11}}; without a label: 0

### Real neurons only the language model flags

**70m:real::56**, MLP neuron(s) 78 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Exponent operator (**) or multiplication (*) inside polynomial expressions in calculus/derivative problems", "Newline or comma at the end of closing lines, such as sign-offs before a name or publisher/address commas in citations", "Spacing and digit tokens in probability problems about picking letters"]}`
- ` contact you to opt in or out.⏎⏎We look forward to receiving your revised manuscript.⏎⏎Kind regards,[[⏎]]⏎Talib Al`
- `).⏎88*c⏎Let g(y) be the first derivative of -y**4/4 + y[[**]]3/3 -`
- `}. What is prob of picking 1 p and 1 z?⏎2/33⏎Calculate prob of picking 2 d and[[ 1]] l when three letters`
- `ddlpv. What is prob of picking 1 u and 1 p?⏎2/39⏎Calculate prob of picking[[ 2]] f when two letters`

**70m:real::60**, MLP neuron(s) 83 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["newline or blank line after a completed statement or line in source code, where a new code block begins", "the auxiliary verb 'was' following a noun-phrase subject in passive scientific prose", "a '+' operator continuing a linear expression in algebra or math"]}`
- `, build_arg_string(kwargs)])⏎                lib.call_module("text", arg_str)[[⏎⏎]]    @fmt_`
- `const Integer code0,⏎    const Integer code1)⏎{⏎    int32 level = start_level;[[⏎⏎]]    while (level`
- `⏎        return⏎    }⏎⏎    let imageViewSnapshot = fromViewTransitionFromView.getViewForTransition()[[⏎⏎]]    let endFrame`
- ` = validRegion⏎        )⏎        assert(sub.requestNext() == expectedMessageWithRegion)⏎    }[[⏎⏎]]    ignore should "`

### Real neurons only the network reader flags

**70m:real::182**, MLP neuron(s) 240 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["the word 'as' introducing a role, capacity, manner, or comparison (e.g. 'as well as', 'as described previously', 'as a function of'), or the role/category noun that directly follows it"]}`
- ` day of action, as they were not behind the call. However, after realizing that they were losing authority among their ranks[[ as]] numerous trade unionists`
- `), they can be used to operate at the nanometric scale on such entities, to assist the formation of ordered particle structures[[ as]] for example [@`
- ` vaguely aware, so conventional. I was to carry Stevens with me into places neither of us could have foreknown, places[[ as]] dense, implac`
- `⏎⏎The truth, as always, lies somewhere in between. While instances certainly exist of companies falsely classing their workers[[ as]]`

**70m:real::197**, MLP neuron(s) 262 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["Semicolons inside C-style for-loop headers (e.g. `for (int i = 0; i < n`), where the next token is the loop condition", "The \"+\" joining terms in polynomial or algebraic simplification expressions, where the next term is predictable", "Math notation tokens before a predictable relation or quantifier target such as `\\in`, `\\ge`, or `\\rightarrow` in set membership and inequality statements", "Boilerplate or fixed-phrase tokens in license, legal, forum, and signature-line text, where the continuation is highly formulaic"]}`
- `)⏎					results.Add(new Vertices(mergeA[i]));⏎⏎				for (int i = 0[[;]] i < mergeB`
- `irectedMultigraph<String> g = new DirectedMultigraph<String>();⏎        for (int i = 0[[;]] i < 10;`
- `:⏎Opinions posted on Free Republic are those of the individual⏎posters and do not necessarily represent the opinion[[ of]] Free Republic or its`
- `) {⏎  size = new unsigned[number_page_segments];⏎  for (unsigned i = 0[[;]] i < number_`

### 50/50 pools the language model calls one job

**70m:pooled:50/50:424-1001**, MLP neuron(s) 543, 1306 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token at a point where the next word is highly predictable from a fixed collocation or elision, such as 'thank' after 'the authors would like to', 'to' after 'grateful'/'indebted', 'Due' opening 'Due to', and French or Italian clitic and elision splits like l[[']] or ha[[i]]"]}`
- ` could be completely inert.⏎⏎Finally, in the spring of 1900, at a meeting in Berlin, Ramsay suggested[[ to]] Mendeleev`
- `.⏎⏎(XLSX)⏎⏎###### ⏎⏎Click here for additional data file.⏎⏎The authors[[ thank]] Katsunori`
- `d**3 + 5937*b*d + 48989*b + 19*d**4 wrt d.[[⏎]]3*b*`
- `udnak a problémáról, és dolgoznak azon, hogy kijavítsák.[[ Az]]t javasol`

**70m:pooled:50/50:420-746**, MLP neuron(s) 538, 970 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Function word or modifier that begins a fixed, highly predictable collocation (e.g., \"motion [for]\", \"represented [as]\", \"very [much]\", \"recommended [that]\", \"various\", \"like\", \"following\", \"respectfully\"), where the next token is largely determined by the phrase"]}`
- `, my name is Katherine Evans and you can mail it to 3217 Union Street." "Yes." "Thank you very[[ much]]." "Miss Evans`
- ` you think of the show?" "Here's the cash." "You can count it all now." "Thank you very[[ much]]." "Two bags`
- ` measures were total time needed to complete the test, number of errors made and number of hesitations. Data were grouped into[[ three]] blocks of 100 words`
- ` dispersion is largely different from that of a liquid crystal cell, and the polymer/cholesteric liquid crystal dispersion has the[[ following]] problems: (1`

## pythia-160m

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 15% | 6% | 8% | 12% | 6% |
| network | 49% | 60% | 52% | 48% | 28% |
| llm | 33% | 62% | 40% | 34% | 16% |

Language-model labels: {'real': {'A': 57, 'B': 10, 'C': 33}, 'pooled 50/50': {'B': 8, 'A': 11, 'C': 31}, 'pooled 70/30': {'C': 20, 'A': 20, 'B': 10}, 'pooled 90/10': {'A': 29, 'C': 17, 'B': 4}, 'related': {'A': 39, 'C': 8, 'B': 3}}; without a label: 0

### Real neurons only the language model flags

**160m:real::266**, MLP neuron(s) 375 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["French pronoun 'en' after reflexive 's' (s'en alla, s'en fermer)", "C/C++ code tokens: identifiers, macros, and comparison operands (GFP_KERNEL, != 0, cleanup calls)", "English function words in mid-clause positions (so that, that, both, in, who)"]}`
- ` donnez pas une meilleure nourriture, dit le Petit-Poucet, je m'[[en]] irai d'`
- `était pas passé une demi-heure que tout était fini et que tous les oiseaux s'[[en]]volèrent de`
- ` following adjustment of the countersink depth. Since the automatic pallet-making machinery does not operate when the chuck is not[[ operating]], pallet making`
- `(pgset, vdp->thread_info, vdp->txn, pgno, &p)) !=[[ 0]])⏎		goto`

**160m:real::447**, MLP neuron(s) 642 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["word-piece fragments that split a word mid-token (e.g., 'quantile', 'coalescent', 'fireplaces', 'ligament', 'linens')", "biomedical and anatomical terms (e.g., 'DNA', 'cGMP', 'MCL', 'testicles', 'mouth', 'cysteine residue')", "'to' after a noun introducing an infinitive purpose clause (e.g., 'a lever to push', 'a tarpaulin to stretch')", "math and code tokens (e.g., 'supremum', 'Vec<char>', 'bitset<8>', '\\partial')"]}`
- ` with a message that the world can — and must — do better.⏎⏎He will look back, placing a wre[[ath]] at the centopath`
- ` striker Mark Sherrod came off early with a knee injury diagnosed by Houston Methodist medical staff as a torn left anterior cruciate[[ ligament]]. The 23-`
- `back}$$ where $\partial ^{\prime }/\partial \zeta $ is taken at constant primed variables and $\partial /\[[partial]] \zeta $ is`
- `varepsilon_m m}\bigg)^2.$$ The second order derivative of $A_j(x)$ can be easily[[ computed]] to be bounded by`

### Real neurons only the network reader flags

**160m:real::333**, MLP neuron(s) 471 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["'of' in partitive 'one of' constructions (e.g., 'one of the best', 'one of Scotland's')", "sentence-initial 'This' referring back to the preceding statement or topic"]}`
- ` **Callanish Standing Stones** , 15 miles west of Stornoway on the A858 road, form one[[ of]] the most complete stone`
- ` If wind behavior near the surface gets your juices flowing, check out the BLIPmap UniViewer online.⏎⏎[[This]] site is a favorite`
- `  %01479-861256; www.glenmorelodge.org.uk)⏎⏎One[[ of]] Britain's leading adventure`
- ` I love all things TV, books, food, music and film as well as going out with friends. This is hopefully[[ somewhere]] you'll be able`

**160m:real::339**, MLP neuron(s) 478 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["the word \"from\" (or a similar preposition like off, on, in, by) right after a verb or noun, introducing a source, origin, or complement"]}`
- ` the field. The analysis of the parallel measurement gave satisfactory agreement for the concentration of the potassium (40 K, as calculated[[ from]] the salinity in`
- ` Fitzgerald into the category of curse-worthy. Larry Fitzgerald’s career high for receiving touchdowns came during the season he appeared[[ on]] the Madden cover`
- ` effects usually described by the concept of color screening ($T\;\gsim\;T_c$) and effects usually described[[ by]] the concept of string`
- `influence on Odling, 82⏎⏎revival of Avogrado's work, 64–65⏎⏎carbon[[,]] 271⏎⏎atomic`

### 50/50 pools the language model calls one job

**160m:pooled:50/50:452-1945**, MLP neuron(s) 652, 2930 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["token inside a formulaic collocation whose next word is highly predictable, e.g. 'it's [time] to', 'at the [time] of', 'for each [set] of', 'such [that]', 'for the [right] amount of time'"]}`
- ` change, the extent of conformational change they undergo is comparable to that observed in induced-fit/other interfaces. For both[[ sets]], points corresponding to`
- ` contrast, confidence intervals obtained with the proposed unbiased estimators are shown in Figure \[fig:unlikely:rg\]. For each[[ value]] of $N$,`
- `⏎Kalix nodded in agreement. She herself had been pursued by their agents many times.⏎⏎"Perhaps it's[[ time]] the clan did something`
- `, casting his eyes on the jewel that hung around Kalix's neck, protecting her from detection. "Maybe it's[[ time]] we took that for`

**160m:pooled:50/50:1471-963**, MLP neuron(s) 2219, 1460 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["a verb or preposition in prose that expects a following object or complement, such as 'subscribing to', 'enroll in', 'refused to', 'prior to joining'"]}`
- ` targetThe number of eligible participants required to be screened in order to reach the recruitment target. Key characteristics and reasons for not[[ joining]] the trial will be`
- ` named Christopher, Julian and Talia. He attended Mercer Island High School, where he played high school soccer prior to[[ joining]] the Sounders Academy`
- `.com/understand/Assistants/UAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Tasks/UDaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Fields[[/]]UEaaaaaaaaaaaaaaaaaaaaaaaa`
- ` said India must seek better market access for goods where India has an edge over ASEAN nations, like leather goods and[[ pharmaceutical]]s, to progress`

## pythia-410m

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 9% | 16% | 12% | 8% | 10% |
| network | 56% | 82% | 64% | 60% | 56% |
| llm | 46% | 64% | 54% | 54% | 38% |

Language-model labels: {'real': {'A': 42, 'C': 46, 'B': 12}, 'pooled 50/50': {'B': 7, 'A': 11, 'C': 32}, 'pooled 70/30': {'C': 27, 'B': 9, 'A': 14}, 'pooled 90/10': {'C': 27, 'A': 20, 'B': 3}, 'related': {'A': 24, 'B': 7, 'C': 19}}; without a label: 0

### Real neurons only the language model flags

**410m:real::76**, MLP neuron(s) 132 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Court citation and case-caption tokens in legal opinions (e.g., 'Tex.Civ.[[App]]', court headers, docket numbers)", "The word 'To' opening the Objective sentence of a medical research abstract after its title", "Identifier continuations in source code (e.g., 'DLT_[[_]]', 'HTMLPurifier[[_]]AttrDef', 'SK[[Action]]')"]}`
- `/7/82⏎⏎1⏎N.D.Tex.⏎⏎AFFIRMED⏎⏎2⏎---------------⏎⏎⏎⏎[[d]] Local Rule 21 case`
- `    only if you received the object code with such an offer, in accord⏎    with subsection 6b.⏎⏎    [[d]]) Convey the`
- `⏎package org.apache.camel.component.file.remote;⏎⏎import org.apache.camel[[.]]Exchange;⏎import`
- ` of Lubbock, 150 S.W.3d 708, 716–17⏎(Tex.[[ App]].—Amarillo`

**410m:real::679**, MLP neuron(s) 1330 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["word-piece fragments that begin a compound or prefixed word, such as 'sub' in sub-net, subradiant, and subdirectory", "the phrase 'during the course of' before 'the trial' or 'commission'", "an opening bracket or parenthesis at the start of a loop condition or grouped expression in code", "a numeric or symbolic token inside an arithmetic or algebra problem, such as a minus sign before a fraction or a denominator"]}`
- `⏎  mutable int a;⏎⏎public:⏎  S2() : a(0) {}⏎};[[⏎]]const S2 b`
- ` have order considerably larger than $O(n)$. Nevertheless, permutations of order $O(n)$, that is, of[[ order]] at most $cn`
- ` referred to as a DUT (Device Under Test)} with a voltage with high precision in order to inspect and test the[[ D]]UT with good precision`
- ` can I make it propagate together with dependencies? I'd like it to work in the way that adding extra_lib as[[ sub]]directory and as a`

### Real neurons only the network reader flags

**410m:real::18**, MLP neuron(s) 31 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["token is a mid-word subword fragment whose continuation completes a word (e.g., 'osing', 'ing', 'ated', 'ikre')"]}`
- `⏎Instagram Lately⏎⏎CT / NY FULL-SERVICE INTERIOR DESIGN⏎⏎FULL-SERV[[ICE]] INTERIOR DESIGN`
- ` more on the exchange). filmed from and followed by Amazon. Please see a friendly UK lead. Please become a college-[[educated]] UK download. H`
- ` Studies in the Physical Sciences_ , 32, 183-196, 2001; M. Gordin, _A Well Or[[dered]] Thing_ , Basic`
- `⏎diesen kleinen Hang hinunter zum⏎echte Skipisten und wenn Sie benutzen⏎dies[[e]] Mülleimer`

**410m:real::113**, MLP neuron(s) 209 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-token or mid-phrase positions in structured text (LaTeX cross-references, RST heading underlines, contractions, code identifiers, CSS class names, and similar) where the next token is largely determined by local syntax or formatting conventions"]}`
- `progress" !global⏎$progress-bar-class: "-bar" !global⏎$progress-bar-padding[[-]]vertical: $base`
- `body"><dd><div class="views-element-container form-group"><div class="view view-profil[[-]]page-concours`
- `⏎                                                     property="doc.role" styleClass="form-control"⏎                                                     data-validation[[-]]help="1(`
- `} (4j-1)!!} \cdot n^{m^2-m}.$$⏎⏎Final comments⏎[[==============]]⏎⏎We have`

### 50/50 pools the language model calls one job

**410m:pooled:50/50:1249-178**, MLP neuron(s) 2496, 335 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Tokens where a comparison, contrast, or coordination construction is set up or completed, such as 'rather', 'not', 'than', 'as', 'and', 'or', and 'to' in 'X rather than Y', 'not X but Y', 'as hard as', or 'and not'"]}`
- ` family. I married a "real Jew" who was himself almost equally divided between a troubled yet ingrained Jewish identity,[[ and]] the pull toward Yan`
- `ness, fairness, peace on earth, and combined the feminine appeal of "The meek shall inherit the earth" with[[ the]] masculine stride of "`
- ` we experience and learn from the world around us.⏎⏎Known as “Event Memory Retrieval and Comparison Theory”[[ or]] EMRC,`
- `;  hnoon-late Wed-Sat, to 6pm Sun), which was voted best bar in the city two[[ years]] running in the Pride`

**410m:pooled:50/50:590-1111**, MLP neuron(s) 1138, 2200 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token right before a pointer to supplementary detail, such as 'for [more information/details/a review/a list]' or 'for example'"]}`
- `closure: this post may contain affiliate links, which won’t change your price but will share some commission. See here[[ for]] more information.⏎`
- `:bickel:2001]). See the book ([-@vdL:robins:2003]) for details and[[ for]] many examples of doubly`
- ` for attributes that are native types (Int, String, Float, etc) can passed as-is (scalar values).[[ Values]] for complex Types (`
- `2. Description of the Related Art⏎Numerous types of handheld electronic devices are known. Examples of such handheld electronic devices[[ include]], for instance,`

## pythia-1b

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 8% | 26% | 18% | 8% | 2% |
| network | 50% | 74% | 68% | 54% | 36% |
| llm | 47% | 64% | 62% | 52% | 18% |

Language-model labels: {'real': {'A': 45, 'C': 47, 'B': 8}, 'pooled 50/50': {'A': 14, 'B': 4, 'C': 32}, 'pooled 70/30': {'C': 31, 'B': 5, 'A': 14}, 'pooled 90/10': {'C': 26, 'A': 20, 'B': 4}, 'related': {'A': 37, 'C': 9, 'B': 4}}; without a label: 0

### Real neurons only the language model flags

**1b:real::40**, MLP neuron(s) 149 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["numeric tokens inside numbers, years, and identifiers", "\"do\" or \"'d\" in the phrase \"How do you...\"", "evaluative adjectives after \"a very\" or \"quite a\" (good, reasonable, honest, great)"]}`
- `        @Override⏎        public int hashCode() {⏎            final int prime = 31;⏎            int result =[[ 1]];⏎            result`
- `⏎4. Self-contained classrooms are not the only places students are segregated⏎⏎A colleague of mine made a very[[ good]] point the other day`
- ` safe to rely on the system's timezone settings. You are *required* to use the date.timezone setting[[ or]] the date_default`
- `itional jump or move depends on uninitialised value(s)⏎==3317==    at 0x8048[[EF]]B: gen_`

**1b:real::86**, MLP neuron(s) 332 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["medical treatment, surgery, and drug terms (treatment, surgical, drugs, cosmetic, repairs, fix)", "action verbs do/doing/did in 'what did you do' style questions", "word-piece fragments split mid-word (urg|ia, Tomas|iello, mic|os, #pragma warning)"]}`
- ` "Carl Blundell?" "The Iron Fist?" "Local wrestler?" "Yeah?" "What did they[[ do]] to him?" "`
- ` motherboard.⏎⏎SOLVED: Does Asrock P4VM8 support GB SATA hard disks? –[[ Fix]]ya⏎⏎Near`
- ` and has inspired hockey fans across Michigan and Ohio. Alden Hirschfeld took the ice this season after recovering from brain[[ surgery]] over six months ago`
- ` I first got my TB & realized how bad the battery was, I went to verizon to find out what I could[[ do]] to make it last`

### Real neurons only the network reader flags

**1b:real::2**, MLP neuron(s) 10 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token where the next token is highly predictable, such as completing a fixed name, phrase, or code idiom (e.g., the 'en' in Kerkkonen, 'son' in Jansson, 'new' before a constructor, '.' before Empty)"]}`
- `are hade hans dotter⏎där borta och då beslutade huset⏎att min oly[[ck]]a och ingen ville`
- `got är helt ny ingen⏎var som att någon säger ja rätt sa jag⏎nej men de vill[[ sk]]juta`
- `               ?????????????????????????????????⏎             var query = repository.GetAll().Select(x =>[[ new]] Functions⏎            {`
- `1], [@pone.0092177-Toth2], wave action [@pone.0092177-Engkv[[ist]]1], [@pone`

**1b:real::202**, MLP neuron(s) 884 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["Subject pronoun ('he', 'it', 'you') right after an introductory location, time, or source phrase (e.g. 'There he', 'In the video it') in narrative or expository prose", "Punctuation or number inside a bibliographic or legal citation (commas after journal or series names, volume or WL numbers, page numbers) where the citation continues"]}`
- ` In Prosser, Law of Torts, 2d ed, chap 7, § 43, p 214, it[[ is]] suggested:⏎"`
- `scale investigation by the Finnish authorities of the prisoner exchange. In the subsequent study by Professor Heikki Ylikangas[[ it]] turned out that about`
- ` was his biggest fan and he was seen on television numerous times. He actually appeared in "Enter the Madkat", where[[ he]] was believed to have`
- ` and M. Sampels, editors, Hybrid metaheuristics , volume 4771 of Lecture Notes in Computer Science[[ ,]] pages 82–95`

### 50/50 pools the language model calls one job

**1b:pooled:50/50:676-1644**, MLP neuron(s) 2770, 6497 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Token inside a fixed multi-token expression (phrasal verbs, split words, compounds, identifiers) where the following token is highly predictable"]}`
- `towards]. As noted, there may be several ways to define such a temporal structure on systems. We refer to related[[ and]] future work for a`
- `', isa => 'Str', required => 1);⏎⏎  use MooseX::ClassAttribute;⏎⏎  class[[_]]has _api_`
- `⏎The world’s best football stage held only once every four years in Football Carnival Slot Machine World Cup Insp[[ired]] Video Slot Online`
- ` what?'⏎⏎We started walking towards the ponds, and had just reached the first of the enclosures when I[[ figured]] out the source of`

**1b:pooled:50/50:1591-660**, MLP neuron(s) 6290, 2708 (MiniLM True, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Token where the next word or symbol is highly predictable from fixed phrasing, boilerplate, or memorized text, such as 'Thank you', 'Splinter Cell', 'typename' in C++ templates, Wikipedia category lines, court-order wording, and arithmetic or number sequences"]}`
- `]⏎public class RRI_I12_AUTHORIZATION_CONTACT : AbstractGroup {⏎⏎	[[///]]<summary> ⏎`
- `(p).⏎-p**4⏎Let o(k) be the second derivative of k**9/680[[40]] + k**8`
- ` this court.⏎        We ORDER Appellant to SHOW CAUSE in writing within fifteen days of the date of⏎this[[ order]] why this appeal should`
- ` area, it appears as text in the mail⏎From: from email⏎Reply-To: reply email ⏎X[[-]]Mailer: PHP`

## pythia-1.4b

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 7% | 4% | 6% | 2% | 6% |
| network | 49% | 74% | 66% | 42% | 38% |
| llm | 47% | 58% | 68% | 44% | 36% |

Language-model labels: {'real': {'C': 47, 'A': 45, 'B': 8}, 'pooled 50/50': {'C': 29, 'A': 14, 'B': 7}, 'pooled 70/30': {'B': 1, 'C': 34, 'A': 15}, 'pooled 90/10': {'C': 22, 'A': 23, 'B': 5}, 'related': {'B': 2, 'C': 18, 'A': 30}}; without a label: 0

### Real neurons only the language model flags

**1.4b:real::202**, MLP neuron(s) 854 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["spatial or positional words such as 'top', 'over', 'roof', 'inside', and 'on top of'", "the 'Pop' fragment in words and acronyms such as Popocatepetl, POPC, Popular, and Populus", "laboratory and apparatus nouns such as 'column', 'membrane', 'plates', and 'slides'"]}`
- ` Knights of the Nine have long been disbanded. Reclaim their former glory as you traverse the far reaches of Cy[[rod]]ill across an epic`
- ` not taken her very long to accept. With a wry smile to their audience of one, she finally laid back on[[ top]] of him so she`
- ` WT plasmid as the template. NS5 expressing plasmids were transformed into *Escherichia coli* strain BL21-Codon[[Plus]](DE3)-`
- `doc.html">Help</a></li>⏎</ul>⏎</div>⏎<div class="sub[[Nav]]">⏎<ul`

**1.4b:real::276**, MLP neuron(s) 1156 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["'else' keyword right after a closing brace in C/C++/JS-style if-blocks", "numeric or sentinel constant (0, 1, -1, NULL) as the operand of a comparison or shift in C-like code", "mid-sentence word or token inside English prose and documentation"]}`
- `orbisSetup_floors(LEBitVector& bv) {⏎  if (bv.noMoreBits[[())]] return False;⏎⏎`
- `a'));⏎        else if (c >= 'A' && c <= 'Z')⏎            bitset |= ([[1]]u << (c`
- `> synthetic_field_trials_;⏎};⏎⏎// Test implementations should be subclasses of an instantiation of[[ this]]⏎// class parameter`
- `NAGED_MEMORY⏎            if (!this->m_exec_conf || !m_is_managed[[)]]⏎                return m`

### Real neurons only the network reader flags

**1.4b:real::293**, MLP neuron(s) 1234 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["mid-sentence token where the next word continues an unfinished clause, such as a temporal or comparative connective (after, before, as, when) or a verb/copula awaiting its complement"]}`
- ` component is the antenna—essentially a transducer that converts time varying electrical current to radiated energy, and often considered a[[ last]] minute addition to the`
- ` fast. Decembrius didn't see it coming. It smashed into the larger werewolf's cheekbone,[[ sending]] him crashing to the`
- ` gorgeous pastel-shaded guestrooms upstairs (single/double £95/130) include a sumptuous breakfast[[ as]] well as sherry`
- `ist and retained a strong interest in chemistry throughout his life, including acting as the editor of _Chemical News_ from[[ when]] he founded it in`

**1.4b:real::414**, MLP neuron(s) 1789 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["Verbs of proving or showing in mathematical proofs (e.g., 'to prove', 'to show', 'sufficient to prove', 'goal is to prove')", "Evaluative adjectives or adverbs of quality, likelihood, or outcome in dialogue (e.g., 'good', 'bad', 'likely', 'big', 'best', 'expected')", "Closing punctuation or parentheses ending a math or code clause"]}`
- ` Indeed, the inclusion $\widetilde{\operatorname{in}(\bb)}\subseteq\operatorname{in}(\tilde\bb)$ is[[ obvious]], and the corresponding`
- `1$.⏎⏎The function $F'(t)$ is equivalent to $F(t)$. \[P1\]⏎[[⏎]]Intuitively, the`
- `]$ of $\bar\nu_m$ and $\hat\nu_m$ respectively, then it is enough to[[ prove]]: $KP_{`
- `. Do you know she was actually wearing a plastic T-shirt?"⏎⏎Thrix winced. It did sound[[ bad]].⏎⏎"`

### 50/50 pools the language model calls one job

**1.4b:pooled:50/50:650-602**, MLP neuron(s) 2665, 2510 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Word or hyphen inside a set phrase or compound where the next token is highly predictable, such as 'very' before 'little/much/own', 'double' before '-decker/-crosser', 'multi' before '-user/-tool', and 'of' in 'Mistress of the Werewolves'"]}`
- ` equation (1):⏎                              θ          beam                =                              sin                          -              1                                ⁡[[                      ]](                                                            `
- ` the team became Midland in ’06. He has since raced in the World Touring Car Championship, part-own[[s]] the Ocean GP2`
- `⏎Cause it's a warrior you seein' here tonight in the flesh⏎I give you somthin' high voltage[[ double]] dosage⏎Runn`
- `ism Speaks goal!⏎$10 suggested donation / $5 for raffle tix⏎⏎May 7 @[[ Double]] Door⏎It's`

**1.4b:pooled:50/50:164-71**, MLP neuron(s) 640, 295 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-sentence pause points where the text breaks off or turns, such as dashes, ellipses, or hedge words like 'sure', 'clear', and 'much' before 'rather', and the next words continue or qualify the clause"]}`
- ` 315 = 387. Suppose c - 9*c + l = 0. Let h(o) = o + 9[[.]] Calculate the least common`
- ` briefcase." "But I couldn't swear about brown." " He came out of the hotel?" " Yeah." "[[Sure]], that figures."`
- ` for cosmic-ray background) and with the exposure rates measured by total gamma-ray counting. From this and other results[[...]]⏎⏎An X`
- ` similar fuel specification), except for the absence of lead. Unofficially, in some aviation fuels industry circles, such usage[[—]]i.e.`

## pythia-2.8b

| Reader | real | pooled 50/50 | pooled 70/30 | pooled 90/10 | related |
|---|---|---|---|---|---|
| minilm | 3% | 10% | 4% | 8% | 4% |
| network | 44% | 50% | 54% | 50% | 40% |
| llm | 46% | 78% | 58% | 46% | 30% |

Language-model labels: {'real': {'A': 42, 'C': 46, 'B': 12}, 'pooled 50/50': {'C': 39, 'A': 7, 'B': 4}, 'pooled 70/30': {'B': 9, 'A': 12, 'C': 29}, 'pooled 90/10': {'C': 23, 'A': 17, 'B': 10}, 'related': {'C': 15, 'A': 29, 'B': 6}}; without a label: 0

### Real neurons only the language model flags

**2.8b:real::122**, MLP neuron(s) 560 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["Tokens inside HTML/code markup, such as closing tags, anchor names, and pre elements", "Mid-sentence tokens in English news and narrative prose, such as verbs like 'declined' and articles before nouns", "Fragments of split words or identifiers, such as 'PLE' in 'PLEASE' and 'Get' in 'HttpGet'", "Japanese and other non-Latin text where a character falls mid-word", "Numeric, citation, and URL fragments"]}`
- ` in huge numbers, and the Z2 Plus picked up a discount recently, bringing the cost of the phone down to �[[�]]�14,999`
- ` ⏎<a name="l00271"></a>00271 ⏎<a name="l00272"></[[a]]>00272 <`
- `type":"video\/quicktime","content_thumb_url":"https:\/\/as1.ftcdn.net[[\/]]jpg\/01\/`
- `173–189, 1997.CrossRef1097-0037\(199705\)29%3A3<173%[[3]]A%3AA`

**2.8b:real::384**, MLP neuron(s) 1799 (MiniLM False, network False, LLM C)

- Language model: `{"label": "C", "patterns": ["mid-word subword fragments in code identifiers and rare words (e.g., 'ob' in observe, 'oc' in mock, 'met' in metacpan)", "verbs or participles that take a complement after a subject, such as 'lodged', 'introduce', 'sneak', 'deemed', 'discovered'", "sentence-final periods and closing punctuation that end a clause or citation", "code-syntax tokens such as '=', '()', and closing brackets in source snippets"]}`
- ` for arm64 (as opposed to just a 64bit runtime) as mentioned in this fix for erikdoe/[[oc]]mock⏎#ifdef`
- `[@pap2]. The analysis of bound states using, for instance, the Schrödinger equation will do better in this respect[[.]] It can, however`
- `To get your bearings, take the hugely entertaining free guided tour with any of the Beefeaters (Yeoman Warders[[).]] Hour-long tours`
- `-old Myanmar national', or 'a 40-year-old Indonesian female'. Maybe that's why the man's name[[ lodged]] in my head –`

### Real neurons only the network reader flags

**2.8b:real::202**, MLP neuron(s) 930 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["Mid-word subword fragments that are completed by following tokens (e.g., 'ots' in Botswana/Cotswolds, 'eps' in Stellar eps-matches, 'ps' in dyspepsia, 'ars' in arsB)", "The particle 'out' after a verb or motion phrase (e.g., 'pull out of', 'headed out', 'radiating out from')"]}`
- ` moderately active (2, 3, 4, and 6), and inactive (1, 7, and 8). The catal[[eps]]y-reversal`
- ` pH monitoring, an abnormal score does not appear to predict response to proton pump inhibitor therapy in patients with unspecified functional dyspe[[ps]]ia.<|endoftext|>The`
- `\big((I\setminus[0,\varepsilon_m])\cap {\ensuremath {\mathbb{R}}}_+\big[[)}{]]m-1},\`
- `M2I10M;mutations=58A,59C,61T⏎seq1	Stellar	[[eps]]-matches	99`

**2.8b:real::372**, MLP neuron(s) 1755 (MiniLM False, network True, LLM B)

- Language model: `{"label": "B", "patterns": ["C-family and Java/C# declaration modifiers and type keywords (static, unsigned, private, bool) at the start of a declaration", "boolean false/FALSE literals in code, config, and typed-value contexts", "the word 'artificial' in scientific prose about artificial seawater or saliva media"]}`
- `out.println("5=" + i + " m=" + m);⏎        }⏎⏎    }⏎⏎    public[[ static]] void main(String`
- `,⏎	dpaa_fd_list,⏎	dpaa_fd_sg⏎};⏎⏎[[static]] inline u64 ld`
- ` declaration, but I'm following your approach:⏎template<class T>⏎struct AnimalTestbench⏎{⏎   [[static]] void TestSay();`
- `⏎⏎    /**⏎     * Used as key for {@link ModelWriterTracker#cache}⏎     */⏎    private[[ static]] class CacheKey {`

### 50/50 pools the language model calls one job

**2.8b:pooled:50/50:1310-1307**, MLP neuron(s) 6607, 6580 (MiniLM False, network True, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-sentence tokens in running English prose, often verbs or function words (of, for, with, in, own, said, told) where the phrase continues into a verb or object"]}`
- `:⏎⏎Blasphemy against Allah: Death⏎⏎Blasphemy against the Prophet Muhammad: Death –[[ even]] if the accuser`
- `*v**3 + 10*v**2 - 1 wrt v?⏎-1788*f**2 -[[ 24]]*f + 24`
- ` requests for Content Objects, and routers across the network route these requests directly through the network to a closest network node that[[ stores]] a copy of the`
- ` with comfy wooden bunks, a convivial living room with peat fire and a welcoming owner who can provide[[ all]] kinds of advice on`

**2.8b:pooled:50/50:1414-1067**, MLP neuron(s) 7165, 5364 (MiniLM False, network False, LLM A)

- Language model: `{"label": "A", "patterns": ["Mid-sentence tokens in running prose or formal text, usually function words, partial words, or line/paragraph breaks, where the sentence or citation is still open and the next token continues the clause"]}`
- ` reconsideration of this Court’s June 27, 2017⏎  order is considered, and it is DENIED, because it[[ does]] not appear that the`
- ` Me and her are tight so I know she will get it and be reading it.⏎⏎Let's see, how[[ has]] this week gone.`
- ` 41, 45-46, 78 S.Ct. 99, 2 L.Ed.2d 80 (1957),[[ that]] "a complaint should`
- `iorari to the United States Court of Appeals for the Fifth Circuit.⏎The petition for writ of certiorari is denied.[[⏎]]Justice BRENNA`
