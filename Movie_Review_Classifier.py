{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FyQRoYokSlqV"
   },
   "source": [
    "# IMPORT LIBRARIES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "FXg5pI7_KUkH"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from nltk.corpus import stopwords \n",
    "from collections import Counter\n",
    "import string\n",
    "import re\n",
    "import seaborn as sns\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "from torch.utils.data import TensorDataset, DataLoader\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "a34nNWliKXZ4",
    "outputId": "18165ba3-e5a6-4cd0-ecad-48a6b9217efa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GPU not available, CPU used\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/shubham/anaconda3/lib/python3.8/site-packages/torch/cuda/__init__.py:52: UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /opt/conda/conda-bld/pytorch_1607370172916/work/c10/cuda/CUDAFunctions.cpp:100.)\n",
      "  return torch._C._cuda_getDeviceCount() > 0\n"
     ]
    }
   ],
   "source": [
    "is_cuda = torch.cuda.is_available()\n",
    "\n",
    "# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.\n",
    "if is_cuda:\n",
    "    device = torch.device(\"cuda\")\n",
    "    print(\"GPU is available\")\n",
    "else:\n",
    "    device = torch.device(\"cpu\")\n",
    "    print(\"GPU not available, CPU used\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "E0yMWRNqKnnG"
   },
   "source": [
    "# LOAD IN AND VISUALIZE THE DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "kmTm-NADKjEA",
    "outputId": "2d30e88e-6daf-4bbb-b9f5-437be0a99a79"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[\"One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side.\"\n",
      " 'A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. <br /><br />The actors are extremely well chosen- Michael Sheen not only \"has got all the polari\" but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams\\' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. A masterful production about one of the great master\\'s of comedy and his life. <br /><br />The realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional \\'dream\\' techniques remains solid then disappears. It plays on our knowledge and our senses, particularly with the scenes concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell\\'s murals decorating every surface) are terribly well done.'\n",
      " 'I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air conditioned theater and watching a light-hearted comedy. The plot is simplistic, but the dialogue is witty and the characters are likable (even the well bread suspected serial killer). While some may be disappointed when they realize this is not Match Point 2: Risk Addiction, I thought it was proof that Woody Allen is still fully in control of the style many of us have grown to love.<br /><br />This was the most I\\'d laughed at one of Woody\\'s comedies in years (dare I say a decade?). While I\\'ve never been impressed with Scarlet Johanson, in this she managed to tone down her \"sexy\" image and jumped right into a average, but spirited young woman.<br /><br />This may not be the crown jewel of his career, but it was wittier than \"Devil Wears Prada\" and more interesting than \"Superman\" a great comedy to go see with friends.'\n",
      " \"Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.\"\n",
      " 'Petter Mattei\\'s \"Love in the Time of Money\" is a visually stunning film to watch. Mr. Mattei offers us a vivid portrait about human relations. This is a movie that seems to be telling us what money, power and success do to people in the different situations we encounter. <br /><br />This being a variation on the Arthur Schnitzler\\'s play about the same theme, the director transfers the action to the present time New York where all these different characters meet and connect. Each one is connected in one way, or another to the next person, but no one seems to know the previous point of contact. Stylishly, the film has a sophisticated luxurious look. We are taken to see how these people live and the world they live in their own habitat.<br /><br />The only thing one gets out of all these souls in the picture is the different stages of loneliness each one inhabits. A big city is not exactly the best place in which human relations find sincere fulfillment, as one discerns is the case with most of the people we encounter.<br /><br />The acting is good under Mr. Mattei\\'s direction. Steve Buscemi, Rosario Dawson, Carol Kane, Michael Imperioli, Adrian Grenier, and the rest of the talented cast, make these characters come alive.<br /><br />We wish Mr. Mattei good luck and await anxiously for his next work.'\n",
      " 'Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it\\'s not preachy or boring. It just never gets old, despite my having seen it some 15 or more times in the last 25 years. Paul Lukas\\' performance brings tears to my eyes, and Bette Davis, in one of her very few truly sympathetic roles, is a delight. The kids are, as grandma says, more like \"dressed-up midgets\" than children, but that only makes them more fun to watch. And the mother\\'s slow awakening to what\\'s happening in the world and under her own roof is believable and startling. If I had a dozen thumbs, they\\'d all be \"up\" for this movie.'\n",
      " \"I sure would like to see a resurrection of a up dated Seahunt series with the tech they have today it would bring back the kid excitement in me.I grew up on black and white TV and Seahunt with Gunsmoke were my hero's every week.You have my vote for a comeback of a new sea hunt.We need a change of pace in TV and this would work for a world of under water adventure.Oh by the way thank you for an outlet like this to view many viewpoints about TV and the many movies.So any ole way I believe I've got what I wanna say.Would be nice to read some more plus points about sea hunt.If my rhymes would be 10 lines would you let me submit,or leave me out to be in doubt and have me to quit,If this is so then I must go so lets do it.\"\n",
      " \"This show was an amazing, fresh & innovative idea in the 70's when it first aired. The first 7 or 8 years were brilliant, but things dropped off after that. By 1990, the show was not really funny anymore, and it's continued its decline further to the complete waste of time it is today.<br /><br />It's truly disgraceful how far this show has fallen. The writing is painfully bad, the performances are almost as bad - if not for the mildly entertaining respite of the guest-hosts, this show probably wouldn't still be on the air. I find it so hard to believe that the same creator that hand-selected the original cast also chose the band of hacks that followed. How can one recognize such brilliance and then see fit to replace it with such mediocrity? I felt I must give 2 stars out of respect for the original cast that made this show such a huge success. As it is now, the show is just awful. I can't believe it's still on the air.\"\n",
      " \"Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake. I've seen 950+ films and this is truly one of the worst of them - it's awful in almost every way: editing, pacing, storyline, 'acting,' soundtrack (the film's only song - a lame country tune - is played no less than four times). The film looks cheap and nasty and is boring in the extreme. Rarely have I been so happy to see the end credits of a film. <br /><br />The only thing that prevents me giving this a 1-score is Harvey Keitel - while this is far from his best performance he at least seems to be making a bit of an effort. One for Keitel obsessives only.\"\n",
      " 'If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.<br /><br />Great Camp!!!'\n",
      " 'Phil the Alien is one of those quirky films where the humour is based around the oddness of everything rather than actual punchlines.<br /><br />At first it was very odd and pretty funny but as the movie progressed I didn\\'t find the jokes or oddness funny anymore.<br /><br />Its a low budget film (thats never a problem in itself), there were some pretty interesting characters, but eventually I just lost interest.<br /><br />I imagine this film would appeal to a stoner who is currently partaking.<br /><br />For something similar but better try \"Brother from another planet\"'\n",
      " \"I saw this movie when I was about 12 when it came out. I recall the scariest scene was the big bird eating men dangling helplessly from parachutes right out of the air. The horror. The horror.<br /><br />As a young kid going to these cheesy B films on Saturday afternoons, I still was tired of the formula for these monster type movies that usually included the hero, a beautiful woman who might be the daughter of a professor and a happy resolution when the monster died in the end. I didn't care much for the romantic angle as a 12 year old and the predictable plots. I love them now for the unintentional humor.<br /><br />But, about a year or so later, I saw Psycho when it came out and I loved that the star, Janet Leigh, was bumped off early in the film. I sat up and took notice at that point. Since screenwriters are making up the story, make it up to be as scary as possible and not from a well-worn formula. There are no rules.\"\n",
      " 'So im not a big fan of Boll\\'s work but then again not many are. I enjoyed his movie Postal (maybe im the only one). Boll apparently bought the rights to use Far Cry long ago even before the game itself was even finsished. <br /><br />People who have enjoyed killing mercs and infiltrating secret research labs located on a tropical island should be warned, that this is not Far Cry... This is something Mr Boll have schemed together along with his legion of schmucks.. Feeling loneley on the set Mr Boll invites three of his countrymen to play with. These players go by the names of Til Schweiger, Udo Kier and Ralf Moeller.<br /><br />Three names that actually have made them selfs pretty big in the movie biz. So the tale goes like this, Jack Carver played by Til Schweiger (yes Carver is German all hail the bratwurst eating dudes!!) However I find that Tils acting in this movie is pretty badass.. People have complained about how he\\'s not really staying true to the whole Carver agenda but we only saw carver in a first person perspective so we don\\'t really know what he looked like when he was kicking a**.. <br /><br />However, the storyline in this film is beyond demented. We see the evil mad scientist Dr. Krieger played by Udo Kier, making Genetically-Mutated-soldiers or GMS as they are called. Performing his top-secret research on an island that reminds me of \"SPOILER\" Vancouver for some reason. Thats right no palm trees here. Instead we got some nice rich lumberjack-woods. We haven\\'t even gone FAR before I started to CRY (mehehe) I cannot go on any more.. If you wanna stay true to Bolls shenanigans then go and see this movie you will not be disappointed it delivers the true Boll experience, meaning most of it will suck.<br /><br />There are some things worth mentioning that would imply that Boll did a good work on some areas of the film such as some nice boat and fighting scenes. Until the whole cromed/albino GMS squad enters the scene and everything just makes me laugh.. The movie Far Cry reeks of scheisse (that\\'s poop for you simpletons) from a fa,r if you wanna take a wiff go ahead.. BTW Carver gets a very annoying sidekick who makes you wanna shoot him the first three minutes he\\'s on screen.'\n",
      " \"The cast played Shakespeare.<br /><br />Shakespeare lost.<br /><br />I appreciate that this is trying to bring Shakespeare to the masses, but why ruin something so good.<br /><br />Is it because 'The Scottish Play' is my favorite Shakespeare? I do not know. What I do know is that a certain Rev Bowdler (hence bowdlerization) tried to do something similar in the Victorian era.<br /><br />In other words, you cannot improve perfection.<br /><br />I have no more to write but as I have to write at least ten lines of text (and English composition was never my forte I will just have to keep going and say that this movie, as the saying goes, just does not cut it.\"\n",
      " \"This a fantastic movie of three prisoners who become famous. One of the actors is george clooney and I'm not a fan but this roll is not bad. Another good thing about the movie is the soundtrack (The man of constant sorrow). I recommand this movie to everybody. Greetings Bart\"\n",
      " \"Kind of drawn in by the erotic scenes, only to realize this was one of the most amateurish and unbelievable bits of film I've ever seen. Sort of like a high school film project. What was Rosanna Arquette thinking?? And what was with all those stock characters in that bizarre supposed Midwest town? Pretty hard to get involved with this one. No lessons to be learned from it, no brilliant insights, just stilted and quite ridiculous (but lots of skin, if that intrigues you) videotaped nonsense....What was with the bisexual relationship, out of nowhere, after all the heterosexual encounters. And what was with that absurd dance, with everybody playing their stereotyped roles? Give this one a pass, it's like a million other miles of bad, wasted film, money that could have been spent on starving children or Aids in Africa.....\"\n",
      " \"Some films just simply should not be remade. This is one of them. In and of itself it is not a bad film. But it fails to capture the flavor and the terror of the 1963 film of the same title. Liam Neeson was excellent as he always is, and most of the cast holds up, with the exception of Owen Wilson, who just did not bring the right feel to the character of Luke. But the major fault with this version is that it strayed too far from the Shirley Jackson story in it's attempts to be grandiose and lost some of the thrill of the earlier film in a trade off for snazzier special effects. Again I will say that in and of itself it is not a bad film. But you will enjoy the friction of terror in the older version much more.\"\n",
      " \"This movie made it into one of my top 10 most awful movies. Horrible. <br /><br />There wasn't a continuous minute where there wasn't a fight with one monster or another. There was no chance for any character development, they were too busy running from one sword fight to another. I had no emotional attachment (except to the big bad machine that wanted to destroy them) <br /><br />Scenes were blatantly stolen from other movies, LOTR, Star Wars and Matrix. <br /><br />Examples<br /><br />>The ghost scene at the end was stolen from the final scene of the old Star Wars with Yoda, Obee One and Vader. <br /><br />>The spider machine in the beginning was exactly like Frodo being attacked by the spider in Return of the Kings. (Elijah Wood is the victim in both films) and wait......it hypnotizes (stings) its victim and wraps them up.....uh hello????<br /><br />>And the whole machine vs. humans theme WAS the Matrix..or Terminator.....<br /><br />There are more examples but why waste the time? And will someone tell me what was with the Nazi's?!?! Nazi's???? <br /><br />There was a juvenile story line rushed to a juvenile conclusion. The movie could not decide if it was a children's movie or an adult movie and wasn't much of either. <br /><br />Just awful. A real disappointment to say the least. Save your money.\"\n",
      " 'I remember this film,it was the first film i had watched at the cinema the picture was dark in places i was very nervous it was back in 74/75 my Dad took me my brother & sister to Newbury cinema in Newbury Berkshire England. I recall the tigers and the lots of snow in the film also the appearance of Grizzly Adams actor Dan Haggery i think one of the tigers gets shot and dies. If anyone knows where to find this on DVD etc please let me know.The cinema now has been turned in a fitness club which is a very big shame as the nearest cinema now is 20 miles away, would love to hear from others who have seen this film or any other like it.'\n",
      " \"An awful film! It must have been up against some real stinkers to be nominated for the Golden Globe. They've taken the story of the first famous female Renaissance painter and mangled it beyond recognition. My complaint is not that they've taken liberties with the facts; if the story were good, that would perfectly fine. But it's simply bizarre -- by all accounts the true story of this artist would have made for a far better film, so why did they come up with this dishwater-dull script? I suppose there weren't enough naked people in the factual version. It's hurriedly capped off in the end with a summary of the artist's life -- we could have saved ourselves a couple of hours if they'd favored the rest of the film with same brevity.\"\n",
      " \"After the success of Die Hard and it's sequels it's no surprise really that in the 1990s, a glut of 'Die Hard on a .....' movies cashed in on the wrong guy, wrong place, wrong time concept. That is what they did with Cliffhanger, Die Hard on a mountain just in time to rescue Sly 'Stop or My Mom Will Shoot' Stallone's career.<br /><br />Cliffhanger is one big nit-pickers dream, especially to those who are expert at mountain climbing, base-jumping, aviation, facial expressions, acting skills. All in all it's full of excuses to dismiss the film as one overblown pile of junk. Stallone even managed to get out-acted by a horse! However, if you an forget all the nonsense, it's actually a very lovable and undeniably entertaining romp that delivers as plenty of thrills, and unintentionally, plenty of laughs.<br /><br />You've got to love John Lithgows sneery evilness, his tick every box band of baddies, and best of all, the permanently harassed and hapless 'turncoat' agent, Rex Linn as Travers.<br /><br />He may of been Henry in 'Portrait of a Serial Killer' but Michael Rooker is noteworthy for a cringe-worthy performance as Hal, he insists on constantly shrieking in painful disbelief at his captors 'that man never hurt anybody' And whilst he surely can't be, it really does look like Ralph Waite's Frank character is grinning as the girl plummets to her death.<br /><br />Mention too must go to former 'London's Burning' actor Craig Fairbrass as the Brit bad guy, who comes a cropper whilst using Hal as a Human Football, yes, you can't help enjoy that bit, Hal needed a good kicking.<br /><br />So forget your better judgement, who cares if 'that could never happen', lower your acting expectations, turn up the volume and enjoy! And if you're looking for Qaulen, he's the one wearing the helicopter.\"\n",
      " 'I had the terrible misfortune of having to view this \"b-movie\" in it\\'s entirety.<br /><br />All I have to say is--- save your time and money!!! This has got to be the worst b-movie of all time, it shouldn\\'t even be called a b-movie, more like an f-movie! Because it fails in all aspects that make a good movie: the story is not interesting at all, all of the actors are paper-thin and not at all believable, it has bad direction and the action sequences are so fake it\\'s almost funny.......almost.<br /><br />The movie is just packed full of crappy one-liners that no respectable person could find amusing in the least little bit.<br /><br />This movie is supposed to be geared towards men, but all the women in it are SO utterly unattractive, especially that old wrinkled thing that comes in towards the end. They try to appear sexy in those weird, horrible costumes and they fail miserably!!!<br /><br />Even some of the most ridiculous b-movies will still give you some laughs, but this is just too painful to watch!!'\n",
      " \"What an absolutely stunning movie, if you have 2.5 hrs to kill, watch it, you won't regret it, it's too much fun! Rajnikanth carries the movie on his shoulders and although there isn't anything more other than him, I still liked it. The music by A.R.Rehman takes time to grow on you but after you heard it a few times, you really start liking it.\"\n",
      " 'First of all, let\\'s get a few things straight here: a) I AM an anime fan- always has been as a matter of fact (I used to watch Speed Racer all the time in Preschool). b) I DO like several B-Movies because they\\'re hilarious. c) I like the Godzilla movies- a lot.<br /><br />Moving on, when the movie first comes on, it seems like it\\'s going to be your usual B-movie, down to the crappy FX, but all a sudden- BOOM! the anime comes on! This is when the movie goes WWWAAAAAYYYYY downhill.<br /><br />The animation is VERY bad & cheap, even worse than what I remember from SPEED RACER, for crissakes! In fact, it\\'s so cheap, one of the few scenes from the movie I \"vividly\" remember is when a bunch of kids run out of a school... & it\\'s the same kids over & over again! The FX are terrible, too; the dinosaurs look worse than Godzilla. In addition, the transition to live action to animation is unorganized, the dialogue & voices(especially the English dub that I viewed) was horrid & I was begging my dad to take the tape out of the DVD/ VHS player; The only thing that kept me surviving was cracking out jokes & comments like the robots & Joel/Mike on MST3K (you pick the season). Honestly, this is the only way to barely enjoy this movie & survive it at the same time.<br /><br />Heck, I\\'m planning to show this to another fellow otaku pal of mine on Halloween for a B-Movie night. Because it\\'s stupid, pretty painful to watch & unintentionally hilarious at the same time, I\\'m giving this movie a 3/10, an improvement from the 0.5/10 I was originally going to give it.<br /><br />(According to my grading scale: 3/10 means Pretty much both boring & bad. As fun as counting to three unless you find a way to make fun of it, then it will become as fun as counting to 15.)'\n",
      " \"This was the worst movie I saw at WorldFest and it also received the least amount of applause afterwards! I can only think it is receiving such recognition based on the amount of known actors in the film. It's great to see J.Beals but she's only in the movie for a few minutes. M.Parker is a much better actress than the part allowed for. The rest of the acting is hard to judge because the movie is so ridiculous and predictable. The main character is totally unsympathetic and therefore a bore to watch. There is no real emotional depth to the story. A movie revolving about an actor who can't get work doesn't feel very original to me. Nor does the development of the cop. It feels like one of many straight-to-video movies I saw back in the 90s ... And not even a good one in those standards.<br /><br />\"\n",
      " \"The Karen Carpenter Story shows a little more about singer Karen Carpenter's complex life. Though it fails in giving accurate facts, and details.<br /><br />Cynthia Gibb (portrays Karen) was not a fine election. She is a good actress , but plays a very naive and sort of dumb Karen Carpenter. I think that the role needed a stronger character. Someone with a stronger personality.<br /><br />Louise Fletcher role as Agnes Carpenter is terrific, she does a great job as Karen's mother.<br /><br />It has great songs, which could have been included in a soundtrack album. Unfortunately they weren't, though this movie was on the top of the ratings in USA and other several countries\"\n",
      " '\"The Cell\" is an exotic masterpiece, a dizzying trip into not only the vast mind of a serial killer, but also into one of a very talented director. This is conclusive evidence of what can be achieved if human beings unleash their uninhibited imaginations. This is boldness at work, pushing aside thoughts to fall into formulas and cliches and creating something truly magnificent. This is the best movie of the year to date.<br /><br />I\\'ve read numerous complaints about this film, anywhere from all style and no substance to poorly cast characters and bad acting. To negatively criticize this film is to miss the point. This movie may be a landmark, a tradition where future movies will hopefully follow. \"The Cell\" has just opened the door to another world of imagination. So can we slam the door in its face and tell it and its director Tarsem Singh that we don\\'t want any more? Personally, I would more than welcome another movie by Tarsem, and would love to see someone try to challenge him.<br /><br />We\\'ve all heard talk about going inside the mind of a serial killer, and yes, I do agree that the \"genre\" is a bit overworked. The 90s were full of movies trying to depict what makes serial killers tick; some of them worked, but most failed. But \"The Cell\" does not blaze down the same trail, we are given a new twist, we are physically transported into the mind and presented with nothing less than a fascinating journey of the most mysterious subject matter ever studied.<br /><br />I like how the movie does not bog us down with too much scientific jargon trying to explain how Jennifer Lopez actually gets to enter the brain of another. Instead, she just lies down on a laboratory table and is wrapped with what looks like really long Twizzlers and jaunted into another entity. \"The Cell\" wants to let you \"see\" what it\\'s all about and not \"how\" it\\'s all about, and I guess that\\'s what some people don\\'t like. True, I do like explanations with my movies, but when a movie ventures onto new ground you must let it do what it desires and simply take it in.<br /><br />I noticed how the film was very dark when it showed reality, maybe to contrast the bright visuals when inside the brain of another. Nonetheless, the set design was simply astonishing. I wouldn\\'t be surprised if this film took home a few Oscars in cinematography, best costumes, best director and the like. If it were up to me it\\'d at least get nominated for best picture.<br /><br />I\\'ve noticed that I\\'ve kind of been repeating myself. Not because there\\'s nothing else to say, but because I can\\'t stress enough how fantastic I thought \"The Cell\" was. If you walk into the movie with a very open mind and to have it taken over with wonders and an eye-popping feast then you are assured a good time. I guess this film was just a little too much for some people, writing it off as \"weird\" or \"crazy\". I am very much into psychology and the imagination of the human mind, so it was right down my alley. Leaving the theater, I heard one audience member say \"Whoever made that movie sure did a lot of good drugs.\" If so, I want what he was smoking.<br /><br />**** (out of 4)'\n",
      " 'This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo... the list goes on and on. It failed miserably at all of them, but there was enough interest to keep me from turning it off until the end.<br /><br />Although I appreciate the spirit behind WAR, INC., it depresses me to see such a clumsy effort, especially when it will be taken by its targets to reflect the lack of the existence of a serious critique, rather than simply the poor writing, direction, and production of this particular film.<br /><br />There is a critique to be made about the corporatization of war. But poking fun at it in this way diminishes the true atrocity of what is happening. Reminds me a bit of THREE KINGS, which similarly trivializes a genuine cause for concern.'\n",
      " 'This movie was so frustrating. Everything seemed energetic and I was totally prepared to have a good time. I at least thought I\\'d be able to stand it. But, I was wrong. First, the weird looping? It was like watching \"America\\'s Funniest Home Videos\". The damn parents. I hated them so much. The stereo-typical Latino family? I need to speak with the person responsible for this. We need to have a talk. That little girl who was always hanging on someone? I just hated her and had to mention it. Now, the final scene transcends, I must say. It\\'s so gloriously bad and full of badness that it is a movie of its own. What crappy dancing. Horrible and beautiful at once.'\n",
      " '\\'War movie\\' is a Hollywood genre that has been done and redone so many times that clichéd dialogue, rehashed plot and over-the-top action sequences seem unavoidable for any conflict dealing with large-scale combat. Once in a while, however, a war movie comes along that goes against the grain and brings a truly original and compelling story to life on the silver screen. The Civil War-era \"Cold Mountain,\" starring Jude Law, Nicole Kidman and Renée Zellweger is such a film.<br /><br />Then again, calling Cold Mountain\" a war movie is not entirely accurate. True enough, the film opens with a (quite literally) quick-and-dirty battle sequence that puts \"Glory\" director Edward Zwick shame. However, \"Cold Mountain\" is not so much about the Civil War itself as it is about the period and the people of the times. The story centers around disgruntled Confederate soldier Inman, played by Jude Law, who becomes disgusted with the gruesome war and homesick for the beautiful hamlet of Cold Mountain, North Carolina and the equally beautiful southern belle he left behind, Ada Monroe, played by Nicole Kidman. At first glance, this setup appears formulaic as the romantic interest back home gives the audience enough sympathy to root for the reluctant soldier\\'s tribulations on the battlefield. Indeed, the earlier segments of the film are relatively unimpressive and even somewhat contrived.<br /><br />\"Cold Mountain\" soon takes a drastic turn, though, as the intrepid hero Inman turns out to be a deserter (incidentally saving the audience from the potentially confusing scenario of wanting to root for the Confederates) and begins a long odyssey homeward. Meanwhile, back at the farm, Ada\\'s cultured ways prove of little use in the fields; soon she is transformed into something of a wilderbeast. Coming to Ada\\'s rescue is the course, tough-as-nails Ruby Thewes, played by Renée Zellweger, who helps Ada put the farm back together and, perhaps more importantly, cope with the loneliness and isolation the war seems to have brought upon Ada.<br /><br />Within these two settings, a vivid, compelling and, at times, very disturbing portrait of the war-torn South unfolds. The characters with whom Inman and Ada interact are surprisingly complex, enhanced by wonderful performances of Brendan Gleeson as Ruby\\'s deadbeat father, Ray Winstone as an unrepentant southern \"lawman,\" and Natalie Portman as a deeply troubled and isolated young mother. All have been greatly affected and changed by \"the war of Northern aggression,\" mostly for the worse. The dark, pervading anti-war message, accented by an effective, haunting score and chillingly beautiful shots of Virginia and North Carolina, is communicated to the audience not so much by gruesome battle scenes as by the scarred land and traumatized people for which the war was fought. Though the weapons and tactics of war itself have changed much in the past century, it\\'s hellish effect on the land is timelessly relevant.<br /><br />Director Anthony Minghella manages to maintain this gloomy mood for most of the film, but the atmosphere is unfortunately denigrated by a rather tepid climax that does little justice to the wonderfully formed characters. The love story between Inman and Ada is awkwardly tacked onto the beginning and end of the film, though the inherently distant, abstracted and even absurd nature of their relationship in a way fits the dismal nature of the rest of the plot.<br /><br />Make no mistake, \"Cold Mountain\" has neither the traits of a feel-good romance nor an inspiring war drama. It is a unique vision of an era that is sure not only to entertain but also to truly absorb the audience into the lives of a people torn apart by a war and entirely desperate to be rid of its terrible repercussions altogether.'\n",
      " 'Taut and organically gripping, Edward Dmytryk\\'s Crossfire is a distinctive suspense thriller, an unlikely \"message\" movie using the look and devices of the noir cycle.<br /><br />Bivouacked in Washington, DC, a company of soldiers cope with their restlessness by hanging out in bars. Three of them end up at a stranger\\'s apartment where Robert Ryan, drunk and belligerent, beats their host (Sam Levene) to death because he happens to be Jewish. Police detective Robert Young investigates with the help of Robert Mitchum, who\\'s assigned to Ryan\\'s outfit. Suspicion falls on the second of the three (George Cooper), who has vanished. Ryan slays the third buddy (Steve Brodie) to insure his silence before Young closes in.<br /><br />Abetted by a superior script by John Paxton, Dmytryk draws precise performances from his three starring Bobs. Ryan, naturally, does his prototypical Angry White Male (and to the hilt), while Mitchum underplays with his characteristic alert nonchalance (his role, however, is not central); Young may never have been better. Gloria Grahame gives her first fully-fledged rendition of the smart-mouthed, vulnerable tramp, and, as a sad sack who\\'s leeched into her life, Paul Kelly haunts us in a small, peripheral role that he makes memorable.<br /><br />The politically engaged Dmytryk perhaps inevitably succumbs to sermonizing, but it\\'s pretty much confined to Young\\'s reminiscence of how his Irish grandfather died at the hands of bigots a century earlier (thus, incidentally, stretching chronology to the limit). At least there\\'s no attempt to render an explanation, however glib, of why Ryan hates Jews (and hillbillies and...).<br /><br />Curiously, Crossfire survives even the major change wrought upon it -- the novel it\\'s based on (Richard Brooks\\' The Brick Foxhole) dealt with a gay-bashing murder. But homosexuality in 1947 was still Beyond The Pale. News of the Holocaust had, however, begun to emerge from the ashes of Europe, so Hollywood felt emboldened to register its protest against anti-Semitism (the studios always quaked at the prospect of offending any potential ticket buyer).<br /><br />But while the change from homophobia to anti-Semitism works in general, the specifics don\\'t fit so smoothly. The victim\\'s chatting up a lonesome, drunk young soldier then inviting him back home looks odd, even though (or especially since) there\\'s a girlfriend in tow. It raises the question whether this scenario was retained inadvertently or left in as a discreet tip-off to the original engine generating Ryan\\'s murderous rage.'\n",
      " '\"Ardh Satya\" is one of the finest film ever made in Indian Cinema. Directed by the great director Govind Nihalani, this one is the most successful Hard Hitting Parallel Cinema which also turned out to be a Commercial Success. Even today, Ardh Satya is an inspiration for all leading directors of India.<br /><br />The film tells the Real-life Scenario of Mumbai Police of the 70s. Unlike any Police of other cities in India, Mumbai Police encompasses a Different system altogether. Govind Nihalani creates a very practical Outlay with real life approach of Mumbai Police Environment.<br /><br />Amongst various Police officers & colleagues, the film describes the story of Anand Velankar, a young hot-blooded Cop coming from a poor family. His father is a harsh Police Constable. Anand himself suffers from his father\\'s ideologies & incidences of his father\\'s Atrocities on his mother. Anand\\'s approach towards immediate action against crime, is an inert craving for his own Job satisfaction. The film is here revolved in a Plot wherein Anand\\'s constant efforts against crime are trampled by his seniors.This leads to frustrations, as he cannot achieve the desired Job-satisfaction. Resulting from the frustrations, his anger is expressed in excessive violence in the remand rooms & bars, also turning him to an alcoholic.<br /><br />The Spirit within him is still alive, as he constantly fights the system. He is aware of the system of the Metro, where the Police & Politicians are a inertly associated by far end. His compromise towards unethical practice is negative. Finally he gets suspended.<br /><br />The Direction is a master piece & thoroughly hard core. One of the best memorable scenes is when Anand breaks in the Underworld gangster Rama Shetty\\'s house to arrest him, followed by short conversation which is fantastic. At many scenes, the film has Hair-raising moments.<br /><br />The Practical approach of Script is a major Punch. Alcoholism, Corruption, Political Influence, Courage, Deceptions all are integral part of Mumbai police even today. Those aspects are dealt brilliantly.<br /><br />Finally, the films belongs to the One man show, Om Puri portraying Anand Velankar traversing through all his emotions absolutely brilliantly.'\n",
      " 'My first exposure to the Templarios & not a good one. I was excited to find this title among the offerings from Anchor Bay Video, which has brought us other cult classics such as \"Spider Baby\". The print quality is excellent, but this alone can\\'t hide the fact that the film is deadly dull. There\\'s a thrilling opening sequence in which the villagers exact a terrible revenge on the Templars (& set the whole thing in motion), but everything else in the movie is slow, ponderous &, ultimately, unfulfilling. Adding insult to injury: the movie was dubbed, not subtitled, as promised on the video jacket.'\n",
      " 'One of the most significant quotes from the entire film is pronounced halfway through by the protagonist, the mafia middle-man Titta Di Girolamo, a physically non-descript, middle-aged man originally from Salerno in Southern Italy. When we\\'re introduced to him at the start of the film, he\\'s been living a non-life in an elegant but sterile hotel in the Italian-speaking Canton of Switzerland for the last ten years, conducting a business we are only gradually introduced to. While this pivotal yet apparently unremarkable scene takes place employees of the the Swiss bank who normally count Di Girolamo\\'s cash tell him that 10,000 dollars are missing from his usual suitcase full of tightly stacked banknotes. At the news, he quietly but icily threatens his coaxing bank manager of wanting to close down his account. Meanwhile he tells us, the spectators, that when you bluff, you have to bluff right through to the end without fear of being caught out or appearing ridiculous. He says: you can\\'t bluff for a while and then halfway through, tell the truth. Having eventually done this - bluffed only halfway through and told the truth, and having accepted the consequences of life and ultimately, love - is exactly the reason behind the beginning of Titta Di Girolamo\\'s troubles. <br /><br />This initially unsympathetic character, a scowling, taciturn, curt man on the verge of 50, a man who won\\'t even reply in kind to chambermaids and waitresses who say hello and goodbye, becomes at one point someone the spectator cares deeply about. At one point in his non-life, Titta decides to feel concern about appearing \"ridiculous\". The first half of the film may be described as \"slow\" by some. It does indeed reveal Di Girolamo\\'s days and nights in that hotel at an oddly disjoined, deliberate pace, revealing seemingly mundane and irrelevant details. However, scenes that may have seemed unnecessary reveal just how essential they are as this masterfully constructed and innovative film unfolds before your eyes. The existence of Titta Di Girolamo - the man with no imagination, identity or life, the unsympathetic character you unexpectedly end up loving and feeling for when you least thought you would - is also conveyed with elegantly edited sequences and very interesting use of music (one theme by the Scottish band Boards of Canada especially stood out). <br /><br />Never was the contrast between the way Hollywood and Italy treat mobsters more at odds than since the release of films such as Le Conseguenze dell\\'Amore or L\\'Imbalsamatore. Another interesting element was the way in which the film made use of the protagonist\\'s insomnia. Not unlike The Machinist (and in a far more explicit way, the Al Pacino film Insomnia), Le Conseguenze dell\\'Amore uses this condition to symbolise a deeper emotional malaise that\\'s been rammed so deep into the obscurity of the unconscious, it\\'s almost impossible to pin-point its cause (if indeed there is one). <br /><br />The young and sympathetic hotel waitress Sofia (played by Olivia Magnani, grand-daughter of the legendary Anna) and the memory of Titta\\'s best friend, a man whom he hasn\\'t seen in 20 years, unexpectedly provide a tiny window onto life that Titta eventually (though tentatively at first) accepts to look through again. Though it\\'s never explicitly spelt out, the spectator KNOWS that to a man like Titta, accepting The Consequences of Love will have unimaginable consequences. A film without a single scene of sex or violence, a film that unfolds in its own time and concedes nothing to the spectator\\'s expectations, Le Conseguenze dell\\'Amore is a fine representative of that small, quiet, discreet Renaissance that has been taking place in Italian cinema since the decline of Cinecittà during the second half of the 70s. The world is waiting for Italy to produce more Il Postino-like fare, more La Vita è Bella-style films... neglecting to explore fine creations like Le Conseguenze dell\\'Amore, L\\'Imbalsamatore and others. Your loss, world.'\n",
      " \"I watched this film not really expecting much, I got it in a pack of 5 films, all of which were pretty terrible in their own way for under a fiver so what could I expect? and you know what I was right, they were all terrible, this movie has a few (and a few is stretching it) interesting points, the occasional camcorder view is a nice touch, the drummer is very like a drummer, i.e damned annoying and, well thats about it actually, the problem is that its just so boring, in what I can only assume was an attempt to build tension, a whole lot of nothing happens and when it does its utterly tedious (I had my thumb on the fast forward button, ready to press for most of the movie, but gave it a go) and seriously is the lead singer of the band that great looking, coz they don't half mention how beautiful he is a hell of a lot, I thought he looked a bit like a meercat, all this and I haven't even mentioned the killer, I'm not even gonna go into it, its just not worth explaining. Anyway as far as I'm concerned Star and London are just about the only reason to watch this and with the exception of London (who was actually quite funny) it wasn't because of their acting talent, I've certainly seen a lot worse, but I've also seen a lot better. Best avoid unless your bored of watching paint dry.\"\n",
      " 'I bought this film at Blockbuster for $3.00, because it sounded interesting (a bit Ranma-esque, with the idea of someone dragging around a skeleton), because there was a cute girl in a mini-skirt on the back, and because there was a Restricted Viewing sticker on it. I thought it was going to be a sweet or at least sincere coming of age story with a weird indie edge. I was 100% wrong.<br /><br />Having watched it, I have to wonder how it got the restricted sticker, since there is hardly any foul language, little violence, and the closest thing to nudity (Honestly! I don\\'t usually go around hoping for it!) is when the girl is in her nightgown and you see her panties (you see her panties a lot in this movie, because no matter what, she\\'s wearing a miniskirt of some sort). Even the anti-religious humor is tame (and lame, caricatured, insincere, derivative, unoriginal, and worst of all not funny in the slightest--it would be better just to listen to Ray Stevens\\' \"Would Jesus Wear a Rolex on His Television Show\"). This would barely qualify as PG-13 (it is Not Rated), but Blockbuster refuses to let anyone under the age of 17 rent this--as if it was pornographic. Any little kid could go in there and rent the edited version of Requiem for a Dream, but they insist that Zack and Reba is worse.<br /><br />It is, but not in that way.<br /><br />In a way, this worries me--the only thing left that could offend people is the idea of the suicide at the beginning. If anybody needs to see movies with honestly portrayed suicides (not this one, but better ones like The Virgin Suicides), it\\'s teenagers. If both of those movies were rated R purely because of the suicide aspect, then I have little chance of turning a story I\\'ve been writing into a PG-13 movie (the main characters are eleven and a half and twelve). Suicide is one of the top three leading causes of death in teenagers (I think it\\'s number 2), so chances are that most teens have been or will be affected by it.<br /><br />Just say no to this movie, though. 2/10.'\n",
      " \"The plot is about the death of little children. Hopper is the one who has to investigate the killings. During the movie it appears that he has some troubles with his daughter. In the end the serial killer get caught. That's it. But before you find out who dunnit, you have to see some terrible acting by all of the actors. It is unbelievable how bad these actors are, including Hopper. I could go on like this but that to much of a waste of my time. Just don't watch the movie. I've warned you.\"\n",
      " \"Ever watched a movie that lost the plot? Well, this didn't even really have one to begin with.<br /><br />Where to begin? The achingly tedious scenes of our heroine sitting around the house with actually no sense of menace or even foreboding created even during the apparently constant thunderstorms (that are strangely never actually heard in the house-great double glazing)? The house that is apparently only a few miles from a town yet is several hours walk away(?) or the third girl who serves no purpose to the plot except to provide a surprisingly quick gory murder just as the tedium becomes unbearable? Or even the beginning which suggests a spate of 20+ killings throughout the area even though it is apparent the killer never ventures far from the house? Or the bizarre ritual with the salt & pepper that pretty much sums up most of the films inherent lack of direction.<br /><br />Add a lead actress who can't act but at least is willing to do some completely irrelevant nude shower scenes and this video is truly nasty, but not in the way you hope.<br /><br />Given a following simply for being banned in the UK in the 80's (mostly because of a final surprisingly over extended murder) it offers nothing but curiosity value- and one classic 'daft' murder (don't worry-its telegraphed at least ten minutes before).<br /><br />After a walk in the woods our victim comes to a rather steep upward slope which they obviously struggle up. Halfway through they see a figure at the top dressed in black and brandishing a large scythe. What do they do? Slide down and run like the rest of us? No, of course not- they struggle to the top and stand conveniently nice and upright in front of the murder weapon.<br /><br />It really IS only a movie as they say..\"\n",
      " \"Okay, so this series kind of takes the route of 'here we go again!' Week in, week out David Morse's character helps out his ride who is in a bit of a pickle - but what's wrong with that!? David Morse is one of the greatest character actors out there, and certainly the coolest, and to have him in a series created by David Koepp - a great writer - is heaven!!<br /><br />Due to the lack of love for this show by many, I can't see it going to a season series - but you never know? The amount of rubbish that has made it beyond that baffles me - let's hope something good can make it past a first series!!!\"\n",
      " 'After sitting through this pile of dung, my husband and I wondered whether it was actually the product of an experiment to see whether a computer program could produce a movie. It was that listless and formulaic. But the U.S. propaganda thrown in your face throughout the film proves--disappointingly--that it\\'s the work of humans. Call me a conspiracy theorist, but quotes like, \"We have to steal the Declaration of Independence to protect it\" seem like ways to justify actions like the invasion of Iraq, etc. The fact that Nicholas Cage spews lines like, \"I would never use the Declaration of Independence as a bargaining chip\" with a straight face made me and my husband wonder whether the entire cast took Valium before shooting each scene. The \"reasoning\" behind each plot turn and new \"clue\" is truly ridiculous and impossible to follow. And there\\'s also a bonus side plot of misogyny, with Dr. Whatever-Her-Name-Was being chided by all involved for \"never shutting up.\" She\\'s clearly in the movie only for looks, but they felt the need to slap a \"Dr.\" title on her character to give her some gravity. At one point, Cage\\'s character says, \"Don\\'t you ever shut up?\" and the camera pans to her looking poutily down at her hands, like she\\'s a child. Truly grotesque. The only benefit to this movie was that it\\'s so astonishingly bad, you do get a few laughs out of it. The really scary thing is that a majority of the people watching the movie with us seemed to enjoy it. Creepy....'\n",
      " \"It had all the clichés of movies of this type and no substance. The plot went nowhere and at the end of the movie I felt like a sucker for watching it. The production was good; however, the script and acting were B-movie quality. The casting was poor because there were good actors mixed in with crumby actors. The good actors didn't hold their own nor did they lift up the others. <br /><br />This movie is not worthy of more words, but I will say more to meet the minimum requirement of ten lines. James Wood and Cuba Gooding, Jr. play caricatures of themselves in other movies. <br /><br />If you are looking for mindless entertainment, I still wouldn't recommend this movie.\"\n",
      " 'This movie is based on the book, \"A Many Splendored Thing\" by Han Suyin and tackles issues of race relations between Asians and Whites, a topic that comes from Han\\'s personal experiences as an Eurasian growing up in China. That background, and the beautiful Hong Kong settings, gives this love story a unique and rather daring atmosphere for its time.<br /><br />Other than that, the story is a stereotypical romance with a memorable song that is perhaps more remembered than the movie itself. The beautiful Jennifer Jones looks the part and gives a wonderful, Oscar nominated performance as a doctor of mixed breed during the advent of Communism in mainland China. William Holden never looked better playing a romantic lead as a journalist covering war torn regions in the world. The acting is top notch, and the chemistry between the two lovers provides for some genuine moments of silver screen affection sure to melt the hearts of those who are romantically inclined.<br /><br />The cinematography really brings out fifty\\'s Hong Kong, especially the hilltop overlooking the harbor where the two lovers spend their most intimate moments. The ending is a real tear-jerker. Some may consider sentimental romances passé, but, for those who enjoy classic Hollywood love stories, this is a shining example.'\n",
      " 'Of all the films I have seen, this one, The Rage, has got to be one of the worst yet. The direction, LOGIC, continuity, changes in plot-script and dialog made me cry out in pain. \"How could ANYONE come up with something so crappy\"? Gary Busey is know for his \"B\" movies, but this is a sure \"W\" movie. (W=waste).<br /><br />Take for example: about two dozen FBI & local law officers surround a trailer house with a jeep wagoneer. Inside the jeep is MA and is \"confused\" as to why all the cops are about. Within seconds a huge gun battle ensues, MA being killed straight off. The cops blast away at the jeep with gary and company blasting away at them. The cops fall like dominoes and the jeep with Gary drives around in circles and are not hit by one single bullet/pellet. MA is killed and gary seems to not to have noticed-damn that guy is tough. Truly a miracle, not since the six-shooter held 300 bullets has there been such a miracle.'\n",
      " 'I had heard good things about \"States of Grace\" and came in with an open mind. I thought that \"God\\'s Army\" was okay, and I thought that maybe Dutcher had improved and matured as a filmmaker. The film began with some shaky acting, and I thought, \"well, maybe it will get better.\" Unfortunately, it never did. The picture starts out by introducing two elders -- Mormon missionaries -- and it seems that the audience will get to know them and grow to care about them. Instead, the story degenerates into a highly improbable series of unfortunate events highlighting blatant disobedience by the missionaries (something that undeniably exists, but rarely on the level that Dutcher portrays) and it becomes almost laughable.<br /><br />Dutcher\\'s only success in this movie is his successful alienation of his target audience. By unrealistically and inaccurately portraying the lives of Mormon missionaries, Dutcher accomplishes nothing more than angering his predominantly Mormon audience. The film in no way reflects reality. Missions are nothing like what Dutcher shows (having served a Mormon mission myself I can attest to this fact) and gang life in California certainly contains much more explicit language than the occasional mild vulgarity.<br /><br />The conclusion, which I\\'m assuming was supposed to touch the audience and inspire them to believe that forgiveness is available to all, was both unbelievable (c\\'mon, the entire mission gathers to see this elder sent home -- and the mom and the girl are standing right next to each other!) and cheesy. Next time, Dutcher, try making a movie that SOMEONE can identify with.'\n",
      " \"This movie struck home for me. Being 29, I remember the '80's and my father working in a factory. I figured, if I worked hard too, if I had pride and never gave up I too could have the American Dream, the house, a few kids, a car all to call my own. I've noted however, without a degree in something (unlike my father that quit at ninth grade) and a keen sense of greed and laziness, you can't get anywhere.<br /><br />I would like to know if anyone has this movie on DVD or VHS. it's made for TV, and I just saw it an hour ago. Ic an't find it anywhere! I'd love to show this to my friends, my pseudo friends, family and other relatives, see what they think and remind them that once upon a time, Americans WOULD work for the sake of feeling honor and that we had pride in what we accomplished!! I think the feeling is still there, but in a heavy downward spiral with so many things being made overseas...\"\n",
      " \"As a disclaimer, I've seen the movie 5-6 times in the last 15 years, and I only just saw the musical this week. This allowed me to judge the movie without being tainted by what was or wasn't in the musical (however, it tainted me when I watched the musical :) ) <br /><br />I actually believe Michael Douglas worked quite well in that role, along with Kasey. I think her 'Let me dance for you scene' is one of the best parts of the movie, a worthwhile addition compared to the musical. The dancers and singing in the movie are much superior to the musical, as well as the cast which is at least 10 times bigger (easier to do in the movie of course). The decors, lighting, dancing, and singing are also much superior in the movie, which should be expected, and was indeed delivered. <br /><br />The songs that were in common with the musical are better done in the movie, the new ones are quite good ones, and the whole movie just delivers more than the musical in my opinion, especially compared to a musical which has few decors. The one bad point on the movie is the obvious cuts between the actors talking, and dubbed singers during the singing portions for some of the characters, but their dancing is impeccable, and the end product was more enjoyable than the musical\"\n",
      " \"Protocol is an implausible movie whose only saving grace is that it stars Goldie Hawn along with a good cast of supporting actors. The story revolves around a ditzy cocktail waitress who becomes famous after inadvertently saving the life of an Arab dignitary. The story goes downhill halfway through the movie and Goldie's charm just doesn't save this movie. Unless you are a Goldie Hawn fan don't go out of your way to see this film.\"\n",
      " 'How this film could be classified as Drama, I have no idea. If I were John Voight and Mary Steenburgen, I would be trying to erase this from my CV. It was as historically accurate as Xena and Hercules. Abraham and Moses got melded into Noah. Lot, Abraham\\'s nephew, Lot, turns up thousands of years before he would have been born. Canaanites wandered the earth...really? What were the scriptwriters thinking? Was it just ignorance (\"I remember something about Noah and animals, and Lot and Canaanites and all that stuff from Sunday School\") or were they trying to offend the maximum number of people on the planet as possible- from Christians, Jews and Muslims, to historians, archaeologists, geologists, psychologists, linguists ...as a matter of fact, did anyone not get offended? Anyone who had even a modicum of taste would have winced at this one!'\n",
      " \"Preston Sturgis' THE POWER AND THE GLORY was unseen by the public for nearly twenty or thirty years until the late 1990s when it resurfaced and even showed up on television. In the meantime it had gained in notoriety because Pauline Kael's THE CITIZEN KANE BOOK had suggested that the Herman Mankiewicz - Orson Welles screenplay for KANE was based on Sturgis' screenplay here. As is mentioned in the beginning of this thread for the film on the IMDb web site, Kael overstated her case.<br /><br />There are about six narrators who take turns dealing with the life of Charles Foster Kane: the newsreel (representing Ralston - the Henry Luce clone), Thatcher's memoirs, Bernstein, Jed Leland, Susan Alexander Kane, and Raymond the butler. Each has his or her different slant on Kane, reflecting their faith or disappointment or hatred of the man. And of course each also reveals his or her own failings when they are telling their version of Kane's story. This method also leads to frequent overlapping re-tellings of the same incident.<br /><br />This is not the situation in THE POWER AND THE GLORY. Yes, like KANE it is about a legendary business leader - here it is Tom Garner (Spencer Tracy), a man who rose from the bottom to being head of the most successful railroad system in the country. But there are only two narrators - they are Garner's right hand man Henry (Ralph Morgan) and his wife (Sarah Padden). This restricts the nearly three dimensional view we get at times of Kane in Garner. Henry, when he narrates, is talking about his boss and friend, whom he respected and loved. His wife is like the voice of the skeptical public - she sees only the flaws in Henry.<br /><br />Typical example: Although he worked his way up, Tom becomes more and more anti-labor in his later years. Unions are troublemakers, and he does not care to be slowed down by their shenanigans. Henry describes Tom's confrontation with the Union in a major walk-out, and how it preoccupied him to the detriment of his home life. But Henry's wife reminds him how Tom used scabs and violence to end the strike (apparently blowing up the Union's headquarters - killing many people). So we have two views of the man but one is pure white and one is pure black.<br /><br />I'm not really knocking THE POWER AND THE GLORY for not duplicating KANE's success (few films do - including all of Orson Welles' other films), but I am aware that the story is presented well enough to hold one's interest to the end. And thanks to the performances of Tracy and Colleen Moore as his wife Sally, the tragedy of the worldly success of the pair is fully brought home.<br /><br />When they marry, Tom wants to do well (in part) to give his wife and their family the benefits he never had. But in America great business success comes at a cost. Tom gets deeply involved with running the railroad empire (he expands it and improves it constantly). But it takes him away from home too much, and he loses touch with Sally. And he also notices Eve (Helen Vinson), the younger woman who becomes his mistress. When Sally learns of his unfaithful behavior it destroys her.<br /><br />Similarly Tom too gets a full shock (which makes him a martyr in the eyes of Henry). Eve marries Tom, and presents him with a son - but it turns out to be Eve's son by Tom's son Tom Jr. (Philip Trent). The discovery of this incestuous cuckolding causes Tom to shoot himself.<br /><br />The film is not a total success - the action jumps at times unconvincingly. Yet it does make the business seem real (note the scene when Tom tells his Board of Directors about his plans to purchase a small rival train line, and he discusses the use of debentures for financing the plans). Sturgis came from a wealthy background, so he could bring in this type of detail. So on the whole it is a first rate film. No CITIZEN KANE perhaps, but of interest to movie lovers as an attempt at business realism with social commentary in Depression America.\"\n",
      " \"Average (and surprisingly tame) Fulci giallo which means it's still quite bad by normal standards, but redeemed by its solid build-up and some nice touches such as a neat time twist on the issues of visions and clairvoyance.<br /><br />The genre's well-known weaknesses are in full gear: banal dialogue, wooden acting, illogical plot points. And the finale goes on much too long, while the denouement proves to be a rather lame or shall I say: limp affair.<br /><br />Fulci's ironic handling of giallo norms is amusing, though. Yellow clues wherever you look.<br /><br />3 out of 10 limping killers\"]\n",
      "\n",
      "['positive' 'positive' 'positive' 'negative' 'positive' 'positive'\n",
      " 'positive' 'negative' 'negative' 'positive' 'negative' 'negative'\n",
      " 'negative' 'negative' 'positive' 'negative' 'positive' 'negative'\n",
      " 'positive' 'negative' 'positive' 'negative' 'positive' 'negative'\n",
      " 'negative' 'positive']\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv('IMDB Dataset.csv')\n",
    "X = df.iloc[ : , 0].values\n",
    "y = df.iloc[ : , 1].values\n",
    "print(X[:50])\n",
    "print()\n",
    "print(y[:26])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "x3-cbCxOSp3c"
   },
   "source": [
    "# DATA PREPROCESSING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "gDIGP9XYKrqV"
   },
   "outputs": [],
   "source": [
    "def preprocess_string(s):\n",
    "    # Remove all non-word characters (everything except numbers and letters)\n",
    "    s = re.sub(r\"[^\\w\\s]\", '', s)\n",
    "    # Replace all runs of whitespaces with no space\n",
    "    s = re.sub(r\"\\s+\", '', s)\n",
    "    # replace digits with no space\n",
    "    s = re.sub(r\"\\d\", '', s)\n",
    "\n",
    "    return s"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "aKoEq61OSuCd"
   },
   "source": [
    "# SPLIT INTO TRAIN AND TEST DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Ydva_7vAPwW8",
    "outputId": "f4b15861-348d-45f8-ad5f-7d6b8e9a91b1"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "shape of train data is (37500,)\n",
      "shape of test data is (12500,)\n"
     ]
    }
   ],
   "source": [
    "X,y = df['review'].values,df['sentiment'].values\n",
    "x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)\n",
    "print(f'shape of train data is {x_train.shape}')\n",
    "print(f'shape of test data is {x_test.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 265
    },
    "id": "fjXkjODvP0E7",
    "outputId": "912402f1-418f-4456-e51c-326e777b152f"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "dd = pd.Series(y_train).value_counts()\n",
    "sns.barplot(x=np.array(['negative','positive']),y=dd.values)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HIgLwqvnS0HE"
   },
   "source": [
    "# TOKENIZATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "eNtF1x_IP2Xg"
   },
   "outputs": [],
   "source": [
    "def preprocess_string(s):\n",
    "    # Remove all non-word characters (everything except numbers and letters)\n",
    "    s = re.sub(r\"[^\\w\\s]\", '', s)\n",
    "    # Replace all runs of whitespaces with no space\n",
    "    s = re.sub(r\"\\s+\", '', s)\n",
    "    # replace digits with no space\n",
    "    s = re.sub(r\"\\d\", '', s)\n",
    "\n",
    "    return s\n",
    "\n",
    "def tockenize(x_train,y_train,x_val,y_val):\n",
    "    word_list = []\n",
    "\n",
    "    stop_words = set(stopwords.words('english')) \n",
    "    for sent in x_train:\n",
    "        for word in sent.lower().split():\n",
    "            word = preprocess_string(word)\n",
    "            if word not in stop_words and word != '':\n",
    "                word_list.append(word)\n",
    "  \n",
    "    corpus = Counter(word_list)\n",
    "    # sorting on the basis of most common words\n",
    "    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]\n",
    "    # creating a dict\n",
    "    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}\n",
    "    \n",
    "    # tockenize\n",
    "    final_list_train,final_list_test = [],[]\n",
    "    for sent in x_train:\n",
    "            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() \n",
    "                                     if preprocess_string(word) in onehot_dict.keys()])\n",
    "    for sent in x_val:\n",
    "            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() \n",
    "                                    if preprocess_string(word) in onehot_dict.keys()])\n",
    "            \n",
    "    encoded_train = [1 if label =='positive' else 0 for label in y_train]  \n",
    "    encoded_test = [1 if label =='positive' else 0 for label in y_val] \n",
    "    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "gtaDR1w5P6-2",
    "outputId": "00a6c379-9a8a-4ba4-f0a2-417d5a6ba3cb"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     /home/shubham/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "<ipython-input-7-d52b72f03a89>:38: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray\n",
      "  return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict\n"
     ]
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "VdUb1EV4P9Cb",
    "outputId": "64d25a75-397e-4e4d-d037-df551031248a"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Length of vocabulary is 1000\n"
     ]
    }
   ],
   "source": [
    "print(f'Length of vocabulary is {len(vocab)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5NghkfWWS4lq"
   },
   "source": [
    "# Analysing review length"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 416
    },
    "id": "oh9WSTc9QUSB",
    "outputId": "d8c1fa57-2ec8-4d61-87ca-c8a30a98ab19"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "count    37500.000000\n",
       "mean        69.008107\n",
       "std         47.876771\n",
       "min          0.000000\n",
       "25%         39.000000\n",
       "50%         54.000000\n",
       "75%         84.000000\n",
       "max        653.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rev_len = [len(i) for i in x_train]\n",
    "pd.Series(rev_len).hist()\n",
    "plt.show()\n",
    "pd.Series(rev_len).describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "A0se8VTwS8ES"
   },
   "source": [
    "# PADDING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "LwadAeyPQZAl"
   },
   "outputs": [],
   "source": [
    "#Now we will pad each of the sequence to max length\n",
    "def padding_(sentences, seq_len):\n",
    "    features = np.zeros((len(sentences), seq_len),dtype=int)\n",
    "    for ii, review in enumerate(sentences):\n",
    "        if len(review) != 0:\n",
    "            features[ii, -len(review):] = np.array(review)[:seq_len]\n",
    "    return features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "_v1CaQuwQcC-"
   },
   "outputs": [],
   "source": [
    "#we have very less number of reviews with length > 500.\n",
    "#So we will consideronly those below it.\n",
    "x_train_pad = padding_(x_train,500)\n",
    "x_test_pad = padding_(x_test,500)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Mc3HabyQTAMK"
   },
   "source": [
    "# Batching and loading as tensor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "Jz1-9jj3Qejc"
   },
   "outputs": [],
   "source": [
    "# create Tensor datasets\n",
    "train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))\n",
    "valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))\n",
    "\n",
    "# dataloaders\n",
    "batch_size = 50\n",
    "\n",
    "# make sure to SHUFFLE your data\n",
    "train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)\n",
    "valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Nzfz9fWeQhGF",
    "outputId": "0d679d79-b60e-481e-87cd-cbaf601b3957"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sample input size:  torch.Size([50, 500])\n",
      "Sample input: \n",
      " tensor([[  0,   0,   0,  ..., 702,   9, 704],\n",
      "        [  0,   0,   0,  ..., 107,   1, 269],\n",
      "        [  0,   0,   0,  ..., 219,  39,   2],\n",
      "        ...,\n",
      "        [  0,   0,   0,  ...,  61,  10,   6],\n",
      "        [  0,   0,   0,  ..., 634,   3, 267],\n",
      "        [  0,   0,   0,  ..., 189, 794,   2]])\n",
      "Sample input: \n",
      " tensor([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0,\n",
      "        0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,\n",
      "        1, 1])\n"
     ]
    }
   ],
   "source": [
    "# obtain one batch of training data\n",
    "dataiter = iter(train_loader)\n",
    "sample_x, sample_y = dataiter.next()\n",
    "\n",
    "print('Sample input size: ', sample_x.size()) # batch_size, seq_length\n",
    "print('Sample input: \\n', sample_x)\n",
    "print('Sample input: \\n', sample_y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "KbZGdb5hTDCM"
   },
   "source": [
    "# MODEL DESIGNING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "KQhOqW5GQjrM"
   },
   "outputs": [],
   "source": [
    "class SentimentRNN(nn.Module):\n",
    "    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):\n",
    "        super(SentimentRNN,self).__init__()\n",
    " \n",
    "        self.output_dim = output_dim\n",
    "        self.hidden_dim = hidden_dim\n",
    " \n",
    "        self.no_layers = no_layers\n",
    "        self.vocab_size = vocab_size\n",
    "    \n",
    "        # embedding and LSTM layers\n",
    "        self.embedding = nn.Embedding(vocab_size, embedding_dim)  \n",
    "        \n",
    "        #lstm\n",
    "        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,num_layers=no_layers, batch_first=True)\n",
    "        \n",
    "        # dropout layer\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "    \n",
    "        # linear and sigmoid layer\n",
    "        self.fc = nn.Linear(self.hidden_dim, output_dim)\n",
    "        self.sig = nn.Sigmoid()\n",
    "        \n",
    "    def forward(self,x,hidden):\n",
    "        batch_size = x.size(0)\n",
    "\n",
    "        # embeddings and lstm_out\n",
    "        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True\n",
    "        #embeds.shape = [50, 500, 1000]\n",
    "        \n",
    "        lstm_out, hidden = self.lstm(embeds, hidden)\n",
    "        \n",
    "        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) \n",
    "        \n",
    "        # dropout and fully connected layer\n",
    "        out = self.dropout(lstm_out)\n",
    "        out = self.fc(out)\n",
    "        \n",
    "        # sigmoid function\n",
    "        sig_out = self.sig(out)\n",
    "        \n",
    "        # reshape to be batch_size first\n",
    "        sig_out = sig_out.view(batch_size, -1)\n",
    "\n",
    "        # get last batch of labels\n",
    "        sig_out = sig_out[:, -1] \n",
    "        \n",
    "        # return last sigmoid output and hidden state\n",
    "        return sig_out, hidden\n",
    "        \n",
    "        \n",
    "        \n",
    "    def init_hidden(self, batch_size):\n",
    "        ''' Initializes hidden state '''\n",
    "\n",
    "        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,\n",
    "        # initialized to zero, for hidden state and cell state of LSTM\n",
    "        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)\n",
    "        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)\n",
    "        hidden = (h0,c0)\n",
    "        return hidden\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ZwWInHLvQqXe",
    "outputId": "1f5cc69a-0a1a-4069-fefc-ff77661a2525"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SentimentRNN(\n",
      "  (embedding): Embedding(1001, 64)\n",
      "  (lstm): LSTM(64, 256, num_layers=2, batch_first=True)\n",
      "  (dropout): Dropout(p=0.3, inplace=False)\n",
      "  (fc): Linear(in_features=256, out_features=1, bias=True)\n",
      "  (sig): Sigmoid()\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "no_layers = 2\n",
    "vocab_size = len(vocab) + 1 #extra 1 for padding\n",
    "embedding_dim = 64\n",
    "output_dim = 1\n",
    "hidden_dim = 256\n",
    "\n",
    "\n",
    "model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)\n",
    "\n",
    "#moving to gpu\n",
    "model.to(device)\n",
    "\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Dmfif2QpTGFR"
   },
   "source": [
    "# TRAINING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "_-3M9pl7Qs_P"
   },
   "outputs": [],
   "source": [
    "#Training\n",
    "\n",
    "# loss and optimization functions\n",
    "lr=0.001\n",
    "\n",
    "criterion = nn.BCELoss()\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n",
    "\n",
    "# function to predict accuracy\n",
    "def acc(pred,label):\n",
    "    pred = torch.round(pred.squeeze())\n",
    "    return torch.sum(pred == label.squeeze()).item()\n",
    "# loss and optimization functions\n",
    "lr=0.001\n",
    "\n",
    "criterion = nn.BCELoss()\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n",
    "\n",
    "# function to predict accuracy\n",
    "def acc(pred,label):\n",
    "    pred = torch.round(pred.squeeze())\n",
    "    return torch.sum(pred == label.squeeze()).item()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ynVTrJKaQwp7",
    "outputId": "413abcf7-d42e-4269-c28e-d0cf3df1cd71"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1\n",
      "train_loss : 0.5049649359782536 val_loss : 0.46645964980125426\n",
      "train_accuracy : 75.59466666666667 val_accuracy : 80.496\n",
      "==================================================\n",
      "Epoch 2\n",
      "train_loss : 0.396049975057443 val_loss : 0.35882036966085434\n",
      "train_accuracy : 82.928 val_accuracy : 84.304\n",
      "==================================================\n",
      "Epoch 3\n",
      "train_loss : 0.3405466283063094 val_loss : 0.3404250083565712\n",
      "train_accuracy : 85.47200000000001 val_accuracy : 85.584\n",
      "==================================================\n",
      "Epoch 4\n",
      "train_loss : 0.3062598773141702 val_loss : 0.325392920255661\n",
      "train_accuracy : 87.232 val_accuracy : 86.2\n",
      "==================================================\n",
      "Epoch 5\n",
      "train_loss : 0.28177479093273483 val_loss : 0.3346347752809525\n",
      "train_accuracy : 88.37333333333333 val_accuracy : 86.168\n",
      "==================================================\n"
     ]
    }
   ],
   "source": [
    "clip = 5\n",
    "epochs = 5 \n",
    "valid_loss_min = np.Inf\n",
    "# train for some number of epochs\n",
    "epoch_tr_loss,epoch_vl_loss = [],[]\n",
    "epoch_tr_acc,epoch_vl_acc = [],[]\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    train_losses = []\n",
    "    train_acc = 0.0\n",
    "    model.train()\n",
    "    # initialize hidden state \n",
    "    h = model.init_hidden(batch_size)\n",
    "    for inputs, labels in train_loader:\n",
    "        \n",
    "        inputs, labels = inputs.to(device), labels.to(device)   \n",
    "        # Creating new variables for the hidden state, otherwise\n",
    "        # we'd backprop through the entire training history\n",
    "        h = tuple([each.data for each in h])\n",
    "        \n",
    "        model.zero_grad()\n",
    "        output,h = model(inputs,h)\n",
    "        \n",
    "        # calculate the loss and perform backprop\n",
    "        loss = criterion(output.squeeze(), labels.float())\n",
    "        loss.backward()\n",
    "        train_losses.append(loss.item())\n",
    "        # calculating accuracy\n",
    "        accuracy = acc(output,labels)\n",
    "        train_acc += accuracy\n",
    "        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.\n",
    "        nn.utils.clip_grad_norm_(model.parameters(), clip)\n",
    "        optimizer.step()\n",
    " \n",
    "    \n",
    "        \n",
    "    val_h = model.init_hidden(batch_size)\n",
    "    val_losses = []\n",
    "    val_acc = 0.0\n",
    "    model.eval()\n",
    "    for inputs, labels in valid_loader:\n",
    "            val_h = tuple([each.data for each in val_h])\n",
    "\n",
    "            inputs, labels = inputs.to(device), labels.to(device)\n",
    "\n",
    "            output, val_h = model(inputs, val_h)\n",
    "            val_loss = criterion(output.squeeze(), labels.float())\n",
    "\n",
    "            val_losses.append(val_loss.item())\n",
    "            \n",
    "            accuracy = acc(output,labels)\n",
    "            val_acc += accuracy\n",
    "            \n",
    "    epoch_train_loss = np.mean(train_losses)\n",
    "    epoch_val_loss = np.mean(val_losses)\n",
    "    epoch_train_acc = train_acc/len(train_loader.dataset)\n",
    "    epoch_val_acc = val_acc/len(valid_loader.dataset)\n",
    "    epoch_tr_loss.append(epoch_train_loss)\n",
    "    epoch_vl_loss.append(epoch_val_loss)\n",
    "    epoch_tr_acc.append(epoch_train_acc)\n",
    "    epoch_vl_acc.append(epoch_val_acc)\n",
    "    print(f'Epoch {epoch+1}') \n",
    "    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')\n",
    "    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')\n",
    "    if epoch_val_loss <= valid_loss_min:\n",
    "        #torch.save()\n",
    "        #print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))\n",
    "        valid_loss_min = epoch_val_loss\n",
    "    print(25*'==')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nXlT97ffTJ3p"
   },
   "source": [
    "# ACCURACY AND LOSS PLOTS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 375
    },
    "id": "9SO_Wfa0Q05u",
    "outputId": "92810c5d-ea7b-47e8-9fd8-92dc090a2207"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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FER0dDTGmLM6Vl5eHkFBQZUUWc3kijZaa8nOziY9PZ2WLVtW67lFRERERESkalXmffmZ0v38qZ3pvXmtGFZWWFhIgwYNXPYDKKdmjKFBgwYuzSKLiIiIiIhI1dB9ee1wpvfmtSI5BOgHsBbQ90hERERERKTu0j1f7XAm36dakxxypezsbOLj44mPj6dx48ZERkYe+Xzo0KGT7rt48WLuvPPO0z5namoqxhi+/PLLMw1bREREREREpE7Izs6mZ8+e1XZfHh0dze7du88m5FqlVtQccrUGDRqQmpoKwGOPPUZgYCD33XffkfXFxcV4eR3/S5mcnExycvJpn3P69On06tWL6dOnM3jw4DMLXERERERERKQOaNCgAT/88ANBQUHVdl/uTtRz6AyNGzeOW2+9le7du/OXv/yFX375hR49epCQkMC5557L+vXrAUhJSWHo0KGAI7F0/fXX069fP1q1asXLL7983GNba/nggw946623mDdvXrmxgv/617/o3LkzcXFxTJgwAYBNmzZx3nnnERcXR2JiIr/99lsVt15ERERERETEtaryvrys559/ntjYWGJjY3nxxRcB2L9/P0OGDCEuLo7Y2Fjee+89ACZMmEDHjh3p0qVLueRVTVfreg79/dPVrMncd8b7l5SU4OnpWW5Zx6b1efTiTqd9rPT0dBYtWoSnpyf79u1jwYIFeHl58fXXX/Pggw/y0Ucf/WGfdevW8e2335KXl0e7du247bbb/jC93KJFi2jZsiWtW7emX79+fP7554waNYovvviCWbNm8fPPP+Pv709OTg4AV155JRMmTGDkyJEUFhaSm5t72m0RERERERERqYizvS8/npp2X37YkiVLmDJlCj///DPWWrp3707fvn357bffaNq0KZ9//jkAubm5ZGdn88knn7Bu3TqMMezdu/e02+Mq6jl0FkaPHn0k0ZSbm8vo0aOJjY3lnnvuYfXq1cfdZ8iQIfj6+hIeHk6jRo3YsWPHH7aZPn06Y8aMAWDMmDFMnz4dgK+//prrrrsOf39/AMLCwsjLyyMjI4ORI0cC4Ofnd2S9iIjImcg/WMyCjbvIPlDq6lCkDsg9UMSyncWuDkNEROqoqrovP2zhwoWMHDmSgIAAAgMDueSSS1iwYAGdO3dm3rx5PPDAAyxYsIDg4GCCg4Px8/Pjhhtu4OOPP65V9+a1rufQ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1zXx4vpGAArTE6kozWJiSSaXDEinR1x4rSoKac8h59w8YF671+5p9/zeM5y7ALj4POPrFM+u2MEL63Zz9zWDKcpK8jocEREREREREQkDiXExTB6UzeRB2QBs3XOodVVR1bJ6/vDyVuJiohjbv3dwY+tMBmYmEVgv03119IbU3d7ewye5r3oN5YW9+ez4/l6HIyIiIiIiIiJhqjCjJ4UZPfnUpYUcOd7C8q37qK5tpKauiQfmruOBuevom9qDy0sCl5+NL0onOSHW67D/ja+KQ845Hl99jBMt8NCNFxMV1b0rdyIiIiIiIiISHhJio7msOIPLijO4G9ix/zCL6pqorm1kzsqdzFhaT0yUMapfWuteRUNyUrrFqiJfFYeeXFrP6r0t3H/dsLDfLEpEREREREREuq++qT2YXl7A9PICjrecZMW2fdTUNVFd28SDf6vlwb/VkpkcH7wDWiYTijM8i9U3xaH6ve/yvbnrGJoexcfHFngdjoiIiIiIiIj4RGx0FGMHpDN2QDp3TRlE4ztHWLRhDzV1TbywbjczX2sgyqB/ShT9hh2if0bXLmjxTXGo+egJSrKT+cSAo91iyZaIiIiIiIiI+FNWSgI3jMrjhlF5tJx0rGzYT01tE3Nf20RWcnyXx+Ob4tCQ3BT+8vlLqamp8ToUEREREREREREAoqOMkQVpjCxIY0TsTnrGd32pJqrLv6OHtGJIRERERERERORUvioOiYiIiIiIiIjIqVQcEhERERERERHxMRWHRERERERERER8TMUhEREREREREREfU3FIRERERERERMTHVBwSEREREREREfExFYdERERERERERHxMxSERERERERERER9TcUhERERERERExMdUHBIRERERERER8TEVh0REREREREREfEzFIRERERERERERHzPnnNcxnMLMmoBtnfgtMoA9nfj+3YEfcgR/5OmHHEF5RhI/5AjKsyP0c85ldtJ7y3no5DmY/mYiix/y9EOO4I88/ZAjKM9I4sn8q9sVhzqbmS13zo32Oo7O5IccwR95+iFHUJ6RxA85gvIUOVd++V1SnpHDDzmCP/L0Q46gPCOJVznqsjIRERERERERER9TcUhERERERERExMf8WBx61OsAuoAfcgR/5OmHHEF5RhI/5AjKU+Rc+eV3SXlGDj/kCP7I0w85gvKMJJ7k6Ls9h0RERERERERE5F/8uHJIRERERERERESCIrI4ZGZTzKzWzDaa2TdOczzezJ4KHl9iZoVdH+WFCyHPT5tZk5m9Efy62Ys4L4SZ/d7MGs1s9RmOm5n9v+DPYJWZjezqGDtCCHlWmNmBNn15T1fHeKHMLN/M/mFma81sjZl96TRtwro/Q8wxEvoywcyWmtnKYJ73naZN2I+zIeYZ9uMsgJlFm9nrZvb8aY6FfV9K19EcrPV42I8NfpiD+WH+BZqDtWkT9v2pOdgpbcJ+nIVuNgdzzkXUFxANbAIGAHHASmBIuzafBx4JPq4EnvI67k7K89PAL7yO9QLzvBwYCaw+w/Grgb8CBlwCLPE65k7KswJ43us4LzDHHGBk8HEyUHea39mw7s8Qc4yEvjQgKfg4FlgCXNKuTSSMs6HkGfbjbDCPO4AnT/e7GQl9qa+u+dIc7JQ2YT82+GEO5of5VzAPzcEipD81BzulTdiPs8E8us0cLBJXDpUDG51zm51zx4AqYFq7NtOAPwYfzwQ+YGbWhTF2hFDyDHvOuUXA2+/TZBrwhAt4FUg1s5yuia7jhJBn2HPO7XLOrQg+PgisA/q2axbW/RlijmEv2D/Nwaexwa/2G9iF/TgbYp5hz8zygGuAx87QJOz7UrqM5mARxA9zMD/Mv0BzsEiiOVhk6W5zsEgsDvUFtrd53sC/DwytbZxzJ4ADQHqXRNdxQskT4Prg0tCZZpbfNaF1qVB/DpFgXHBp5V/NbKjXwVyI4JLIEQT+F6CtiOnP98kRIqAvg0tg3wAagQXOuTP2ZRiPs6HkCeE/zv4MuAs4eYbjEdGX0iU0BztVuI8NZxMxn9lnEfaf2W1pDhb+/ak52CnCfZztVnOwSCwOyb/MAQqdcxcDC/hX1VHCzwqgn3OuDPg58JzH8Zw3M0sCngW+7Jx7x+t4OsNZcoyIvnTOtTjnhgN5QLmZDfM6ps4QQp5hPc6a2YeBRufca17HIhJhwnpskFYR8Zn9Hs3BIqM/NQdrFdbjbHecg0VicWgH0LZqmBd87bRtzCwG6AXs7ZLoOs5Z83TO7XXOHQ0+fQwY1UWxdaVQ+jvsOefeeW9ppXNuHhBrZhkeh3XOzCyWwAf2n51zs07TJOz782w5Rkpfvsc5tx/4BzCl3aFIGGdbnSnPCBhnxwNTzWwrgUtjJpvZ/7RrE1F9KZ1Kc7CgCBgbQhH2n9lnE0mf2ZqDRVZ/guZgETDOdrs5WCQWh5YBxWbW38ziCGzcNLtdm9nAp4KPbwAWOufC7RrGs+bZ7jrhqQSuvY00s4FPWsAlwAHn3C6vg+poZtbnvetLzaycwN9uWA3ywfh/B6xzzv3kDM3Cuj9DyTFC+jLTzFKDj3sAVwLr2zUL+3E2lDzDfZx1zn3TOZfnnCsk8Dmy0Dn38XbNwr4vpctoDhYU7mNDiML6MzsUkfCZDZqDtWkT9v2pOdgpbcJ6nO2Oc7CYznpjrzjnTpjZ7cB8AneT+L1zbo2ZfRdY7pybTWDg+JOZbSSwCV2ldxGfnxDz/KKZTQVOEMjz054FfJ7MbAaBOwtkmFkD8B0CG5LhnHsEmEfg7gobgXeBz3gT6YUJIc8bgP8ysxPAYaAy3AZ5AtXxTwBvBq8fBvgWUAAR05+h5BgJfZkD/NHMoglMrJ52zj0faeMsoeUZ9uPs6URgX0oX0BwsssYGP8zBfDL/As3BIqk/NQeLoHH2dLzsSwu/vwcREREREREREekokXhZmYiIiIiIiIiIhEjFIRERERERERERH1NxSERERERERETEx1QcEhERERERERHxMRWHRERERERERER8TMUhEREREREREREfU3FIRERERERERMTHVBwSEREREREREfGx/w8VonEh9FR6EgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1440x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light",
      "tags": []
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize = (20, 6))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(epoch_tr_acc, label='Train Acc')\n",
    "plt.plot(epoch_vl_acc, label='Validation Acc')\n",
    "plt.title(\"Accuracy\")\n",
    "plt.legend()\n",
    "plt.grid()\n",
    "    \n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(epoch_tr_loss, label='Train loss')\n",
    "plt.plot(epoch_vl_loss, label='Validation loss')\n",
    "plt.title(\"Loss\")\n",
    "plt.legend()\n",
    "plt.grid()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yig0BO1MTNPg"
   },
   "source": [
    "# INFERENCE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "5FNjtGmRSLK8"
   },
   "outputs": [],
   "source": [
    "#Inference\n",
    "def predict_text(text):\n",
    "        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() \n",
    "                         if preprocess_string(word) in vocab.keys()])\n",
    "        word_seq = np.expand_dims(word_seq,axis=0)\n",
    "        pad =  torch.from_numpy(padding_(word_seq,500))\n",
    "        inputs = pad.to(device)\n",
    "        batch_size = 1\n",
    "        h = model.init_hidden(batch_size)\n",
    "        h = tuple([each.data for each in h])\n",
    "        output, h = model(inputs, h)\n",
    "        return(output.item())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "qtEWMJt4SN1J",
    "outputId": "7f05e9a7-4d5c-45f5-e4d9-e37b6a15a082"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\"Ardh Satya\" is one of the finest film ever made in Indian Cinema. Directed by the great director Govind Nihalani, this one is the most successful Hard Hitting Parallel Cinema which also turned out to be a Commercial Success. Even today, Ardh Satya is an inspiration for all leading directors of India.<br /><br />The film tells the Real-life Scenario of Mumbai Police of the 70s. Unlike any Police of other cities in India, Mumbai Police encompasses a Different system altogether. Govind Nihalani creates a very practical Outlay with real life approach of Mumbai Police Environment.<br /><br />Amongst various Police officers & colleagues, the film describes the story of Anand Velankar, a young hot-blooded Cop coming from a poor family. His father is a harsh Police Constable. Anand himself suffers from his father's ideologies & incidences of his father's Atrocities on his mother. Anand's approach towards immediate action against crime, is an inert craving for his own Job satisfaction. The film is here revolved in a Plot wherein Anand's constant efforts against crime are trampled by his seniors.This leads to frustrations, as he cannot achieve the desired Job-satisfaction. Resulting from the frustrations, his anger is expressed in excessive violence in the remand rooms & bars, also turning him to an alcoholic.<br /><br />The Spirit within him is still alive, as he constantly fights the system. He is aware of the system of the Metro, where the Police & Politicians are a inertly associated by far end. His compromise towards unethical practice is negative. Finally he gets suspended.<br /><br />The Direction is a master piece & thoroughly hard core. One of the best memorable scenes is when Anand breaks in the Underworld gangster Rama Shetty's house to arrest him, followed by short conversation which is fantastic. At many scenes, the film has Hair-raising moments.<br /><br />The Practical approach of Script is a major Punch. Alcoholism, Corruption, Political Influence, Courage, Deceptions all are integral part of Mumbai police even today. Those aspects are dealt brilliantly.<br /><br />Finally, the films belongs to the One man show, Om Puri portraying Anand Velankar traversing through all his emotions absolutely brilliantly.\n",
      "======================================================================\n",
      "Actual sentiment is  : positive\n",
      "======================================================================\n",
      "Predicted sentiment is positive with a probability of 0.9123875498771667\n"
     ]
    }
   ],
   "source": [
    "index = 31\n",
    "print(df['review'][index])\n",
    "print('='*70)\n",
    "print(f'Actual sentiment is  : {df[\"sentiment\"][index]}')\n",
    "print('='*70)\n",
    "pro = predict_text(df['review'][index])\n",
    "status = \"positive\" if pro > 0.5 else \"negative\"\n",
    "pro = (1 - pro) if status == \"negative\" else pro\n",
    "print(f'Predicted sentiment is {status} with a probability of {pro}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "VmwEnH1jSQFY",
    "outputId": "f2723dba-3810-4f9c-ced0-8c8cdfc85f4b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "As a disclaimer, I've seen the movie 5-6 times in the last 15 years, and I only just saw the musical this week. This allowed me to judge the movie without being tainted by what was or wasn't in the musical (however, it tainted me when I watched the musical :) ) <br /><br />I actually believe Michael Douglas worked quite well in that role, along with Kasey. I think her 'Let me dance for you scene' is one of the best parts of the movie, a worthwhile addition compared to the musical. The dancers and singing in the movie are much superior to the musical, as well as the cast which is at least 10 times bigger (easier to do in the movie of course). The decors, lighting, dancing, and singing are also much superior in the movie, which should be expected, and was indeed delivered. <br /><br />The songs that were in common with the musical are better done in the movie, the new ones are quite good ones, and the whole movie just delivers more than the musical in my opinion, especially compared to a musical which has few decors. The one bad point on the movie is the obvious cuts between the actors talking, and dubbed singers during the singing portions for some of the characters, but their dancing is impeccable, and the end product was more enjoyable than the musical\n",
      "======================================================================\n",
      "Actual sentiment is  : positive\n",
      "======================================================================\n",
      "predicted sentiment is positive with a probability of 0.8959076404571533\n"
     ]
    }
   ],
   "source": [
    "index = 45\n",
    "print(df['review'][index])\n",
    "print('='*70)\n",
    "print(f'Actual sentiment is  : {df[\"sentiment\"][index]}')\n",
    "print('='*70)\n",
    "pro = predict_text(df['review'][index])\n",
    "status = \"positive\" if pro > 0.5 else \"negative\"\n",
    "pro = (1 - pro) if status == \"negative\" else pro\n",
    "print(f'predicted sentiment is {status} with a probability of {pro}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "wFquIsoLSSkr"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "name": "18NA10018_Assignment-2.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
