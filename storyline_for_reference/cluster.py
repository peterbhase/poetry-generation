import nltk
import gensim
import re
import numpy as np
from nltk.corpus import state_union
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from rake_nltk import Rake
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#import RAKE
import json
import pickle
from gensim.models.keyedvectors import KeyedVectors


filepath = "corpuses/dirty/MovieSummaries/"
#filepath = ""


glove_model = KeyedVectors.load_word2vec_format('models/glove.6B.300d.word2vec.txt', binary=False)


dictionary = [""]

def stemming(words):
    ps = PorterStemmer()
    result = ""
    for word in words.split():
        result = result + ps.stem(word)+" "
    return result

#remove stop word -> keyword extraction & delete repetitions
def extract_keywords(thepath):
    t = ""
#    f1 = open('l1.txt', 'w')
    count = 0
    with open(thepath, mode = 'r', encoding = 'utf-8') as f:
        line = f.readline()
        while line:
            if line in ['\n', '\r\n']:
                line = f.readline()
                continue
            count = count + 1
            line_raw = f.readline()
#            line = re.sub(r'[^\w\s]','',line_raw.lower())
            t = t + line
            line = f.readline()

    r = Rake()
#    Rake_object = RAKE.Rake(RAKE.SmartStopList())
#    dict = Rake_object.run(thepath, 3, 1, 5)
    r.extract_keywords_from_text(t)
    dict = r.get_ranked_phrases()
    print (dict)
    return dict

def extract_keyword_in_cluster(sentence_cluster, rake_list):
    count = -1
    keyword_list = [None, None, None, None]
    for list_member in rake_list:
        if list_member in sentence_cluster:
            return
    return

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def extract_from_line(sentence):

    #run Rake on initial sentence
    r = Rake()
    r.extract_keywords_from_text(sentence)
    rake_result = r.get_ranked_phrases()
    dict = ""
    for item in rake_result:
        dict = dict + item + " "
    dict = re.sub(r'[^\w\s]','',dict)
    print (type(dict))
    word_tokenize_dict = word_tokenize(dict)
    tag_dict = nltk.pos_tag(word_tokenize_dict)

    dict_pos_tagged = ""
    possible_tags = ['CC', 'DT', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'PDT', 'PRP', 'PRP$', 'RB',
                     'RBR', 'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB']
    noun_tags = ['NN', 'NNS', 'PRP', 'PRP$']
    v_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adj_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    none_tags = ['CC', 'DT', 'MD', 'PDT', 'RP', 'WRB']
    for item in tag_dict:
        if item[1] in possible_tags:
            dict_pos_tagged = dict_pos_tagged + item[0] + " "
    dict_pos_tagged = ' '.join(unique_list(dict_pos_tagged.split()))
    print (dict_pos_tagged)
    #do naive ranking? so that repeated words could be better used?

    #run word tokenizer for calculating how many wordss
    sentence_temp = re.sub(r'[^\w\s]','',sentence)
    #this ignores things like "debbie's", but in this case, debbie is unlikely to be chosen as a keyword
    word_tokenize_list = word_tokenize(sentence_temp)
    how_many_words = len(word_tokenize_list)

    if (how_many_words >= 4):
        each_cluster_sentences = (int) (how_many_words / 4)
        remaining = how_many_words % 4
    else:
        return None

    finallist = [[None, None, None, None],
                 [None, None, None, None],
                 [None, None, None, None],
                 [None, None, None, None]]

    wholelist = []
    wholelistcount = []
    for i in range(0, 4):
        wordcount = -1
        ncount = 0
        vcount = 0
        adjcount = 0
        nonecount = 0
        sentencei = ""
        for j in range(i * each_cluster_sentences, (i+1) * each_cluster_sentences):
            sentencei = sentencei + word_tokenize_list[j] + " "
        print (sentencei)
        for item in dict_pos_tagged.split():
            if wordcount == 3:
                break
            if item in sentencei:
                word_tokenize_word = word_tokenize(item)
                tag_dict = nltk.pos_tag(word_tokenize_word)
                if tag_dict[0][1] in noun_tags:
                    if ncount > 2:
                        continue
                if tag_dict[0][1] in v_tags:
                    if vcount > 2:
                        continue
                if tag_dict[0][1] in adj_tags:
                    if adjcount > 2:
                        continue
                if tag_dict[0][1] in none_tags:
                    if nonecount > 2:
                        continue
                if item in wholelist:
                    print ("item in list")
                    index = wholelist.index(item)
                    itemcount = wholelistcount[index]
                    if itemcount > 2:
                        continue
                    else:
                        wholelistcount[index] = itemcount + 1
                        wordcount = wordcount + 1
                        finallist[i][wordcount] = item
                else:
                    wholelist.append(item)
                    wholelistcount.append(1)
                    wordcount = wordcount + 1
                    finallist[i][wordcount] = item
                if tag_dict[0][1] in none_tags:
                    nonecount = nonecount + 1
                if tag_dict[0][1] in v_tags:
                    vcount = vcount + 1
                if tag_dict[0][1] in adj_tags:
                    adjcount = adjcount + 1
                if tag_dict[0][1] in noun_tags:
                    ncount = ncount + 1
    outputlist = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    tempflag = True
    count = -1
    for item in finallist:
        for eachitem in item:
            count = count + 1
            if eachitem == None:
                tempflag = False
                break
            else:
                outputlist[count] = eachitem
        if tempflag == False:
            break
    #print (finallist)
    return (outputlist, tempflag)


def extract_set(total_dict):
    return

    #generate word list with pos_tag
    #randomly select




#extract_pos_tags("Corpuses/dirty/booksummaries.txt")
#s1 = "Debbie's favorite band is Dream Street, and her favorite member is Chris Trousdale. When Chris gets a fever while travelling on the Dream Street tour, in a haze, he strays away and ends up in Debbie's bed, much to the shock of his 'Biggest Fan,' who thinks she's in heaven. Debbie proposes that Chris stay with her and he agrees. So, over the week they spend time together and she secretly hides him so he can escape the pressures of being a pop star for a little while. Chris even attends high school with Debbie, while disguised as a nerd. Meanwhile, the band's managers are going crazy at the loss of the star, thinking he has been kidnapped. At the end of the week Debbie and Chris  go to her high school prom where two jealous popular girls figure out Chris's true identity and tell the police about Chris's whereabouts, splitting him and Debbie up. They are eventually reunited on stage at a concert, ending in a sweet, final kiss and a performance by Dream Street."
#s1 = "The nation of Panem consists of a wealthy Capitol and twelve poorer districts. As punishment for a past rebellion, each district must provide a boy and girl  between the ages of 12 and 18 selected by lottery  for the annual Hunger Games. The tributes must fight to the death in an arena; the sole survivor is rewarded with fame and wealth. In her first Reaping, 12-year-old Primrose Everdeen is chosen from District 12. Her older sister Katniss volunteers to take her place. Peeta Mellark, a baker's son who once gave Katniss bread when she was starving, is the other District 12 tribute. Katniss and Peeta are taken to the Capitol, accompanied by their frequently drunk mentor, past victor Haymitch Abernathy. He warns them about the 'Career' tributes who train intensively at special academies and almost always win. During a TV interview with Caesar Flickerman, Peeta unexpectedly reveals his love for Katniss. She is outraged, believing it to be a ploy to gain audience support, as 'sponsors' may provide in-Games gifts of food, medicine, and tools. However, she discovers Peeta meant what he said. The televised Games begin with half of the tributes killed in the first few minutes; Katniss barely survives ignoring Haymitch's advice to run away from the melee over the tempting supplies and weapons strewn in front of a structure called the Cornucopia. Peeta forms an uneasy alliance with the four Careers. They later find Katniss and corner her up a tree. Rue, hiding in a nearby tree, draws her attention to a poisonous tracker jacker nest hanging from a branch. Katniss drops it on her sleeping besiegers. They all scatter, except for Glimmer, who is killed by the insects. Hallucinating due to tracker jacker venom, Katniss is warned to run away by Peeta. Rue cares for Katniss for a couple of days until she recovers. Meanwhile, the alliance has gathered all the supplies into a pile. Katniss has Rue draw them off, then destroys the stockpile by setting off the mines planted around it. Furious, Cato kills the boy assigned to guard it. As Katniss runs from the scene, she hears Rue calling her name. She finds Rue trapped and releases her. Marvel, a tribute from District 1, throws a spear at Katniss, but she dodges the spear, causing it to stab Rue in the stomach instead. Katniss shoots him dead with an arrow. She then comforts the dying Rue with a song. Afterward, she gathers and arranges flowers around Rue's body. When this is televised, it sparks a riot in Rue's District 11. President Snow summons Seneca Crane, the Gamemaker, to express his displeasure at the way the Games are turning out. Since Katniss and Peeta have been presented to the public as 'star-crossed lovers', Haymitch is able to convince Crane to make a rule change to avoid inciting further riots. It is announced that tributes from the same district can win as a pair. Upon hearing this, Katniss searches for Peeta and finds him with an infected sword wound in the leg. She portrays herself as deeply in love with him and gains a sponsor's gift of soup. An announcer proclaims a feast, where the thing each survivor needs most will be provided. Peeta begs her not to risk getting him medicine. Katniss promises not to go, but after he falls asleep, she heads to the feast. Clove ambushes her and pins her down. As Clove gloats, Thresh, the other District 11 tribute, kills Clove after overhearing her tormenting Katniss about killing Rue. He spares Katniss 'just this time.for Rue'. The medicine works, keeping Peeta mobile. Foxface, the girl from District 5, dies from eating nightlock berries she stole from Peeta; neither knew they are highly poisonous. Crane changes the time of day in the arena to late at night and unleashes a pack of hound-like creatures to speed things up. They kill Thresh and force Katniss and Peeta to flee to the roof of the Cornucopia, where they encounter Cato. After a battle, Katniss wounds Cato with an arrow and Peeta hurls him to the creatures below. Katniss shoots Cato to spare him a prolonged death. With Peeta and Katniss apparently victorious, the rule change allowing two winners is suddenly revoked. Peeta tells Katniss to shoot him. Instead, she gives him half of the nightlock. However, before they can commit suicide, they are hastily proclaimed the victors of the 74th Hunger Games. Haymitch warns Katniss that she has made powerful enemies after her display of defiance. She and Peeta return to District 12, while Crane is locked in a room with a bowl of nightlock berries, and President Snow considers the situation."
#s1 = "Now fourteen years old, Harry Potter dreams of an elderly man, Frank Bryce, who is killed after overhearing Lord Voldemort discussing plans with Peter 'Wormtail' Pettigrew and Barty Crouch Jr. The Quidditch World Cup allows Harry to take his mind off his nightmares until followers of Voldemort known as Death Eaters terrorise the spectators' campsites after the match, and Crouch Jr. summons the Dark Mark, a sign showing that Voldemort is returning to power. At Hogwarts, headmaster Albus Dumbledore introduces ex-Auror Alastor 'Mad-Eye' Moody as the new Defence Against the Dark Arts teacher. In their first Defence Against the Dark Arts lesson, the students learn of the three Unforgivable Curses. The Imperius Curse causes absence of free will, the Cruciatus Curse causes unbearable pain, and the final curse, Avada Kedavra, causes death. Dumbledore announces that the school will host the Triwizard Tournament, in which one wizard from each of the three magical schools competes in three challenges. The champions are selected by the Goblet of Fire, a magical cup into which the candidates' names are placed. Fred and George attempt to enter using an aging potion as no one under 17 can enter. This fails miserably. Cedric Diggory, a student from the House of Hufflepuff, is chosen to represent Hogwarts, Viktor Krum is chosen to represent Durmstrang Institute, and Fleur Delacour is selected to represent Beauxbatons Academy of Magic. The Goblet unexpectedly chooses a fourth champion: Harry. As Harry is underage and should have been ineligible to compete, Hogwarts teachers and students grow suspicious, and the feat drives Ron and Harry apart. The teachers want Dumbledore to pull Harry out of the tournament, but the four champions are bound by a magical contract and therefore Dumbledore has no choice and Harry must compete. For the first task of the Triwizard Tournament, each of the champions must retrieve a golden egg guarded by a dragon. Mad-Eye advises Harry to use his talent for flying to overcome the dragon. Harry enters the first task and summons his broomstick to retrieve the egg, which contains information about the second challenge. The students are soon informed of the Yule Ball, a Christmas Eve ball held during the Triwizard Tournament. Ron and Harry have trouble finding dates to the ball and when they find out that Hermione is attending with Viktor Krum, Ron becomes jealous. In exchange for previous aid, Cedric provides Harry with a clue that prompts him to open the egg underwater. With help from Moaning Myrtle, he learns that the second task entails the retrieval of 'something precious' to each of the competitors from the nearby Black Lake, where there are mermaids. While preparing for the task, Neville Longbottom provides Harry with Gillyweed, enabling him to breathe underwater for one hour. Harry is the first to arrive at the location, and finds Ron, Hermione, Cho Chang and Fleur's sister, Gabrielle, in suspended animation. Finishing last after attempting to free Ron and Gabrielle, Harry is awarded second place for 'outstanding moral fiber', behind Cedric. Following an exchange with Mad-Eye, Ministry official Barty Crouch, Sr. is found dead by Harry shortly after the second task. While waiting in Dumbledore's office, Harry's curiosity leads him to look into Dumbledore's pensieve, causing him to revisit one of Dumbledore's memories. He witnesses a trial before the Wizengamot in which captured Death Eater Igor Karkaroff, the current headmaster of Durmstrang, denounces a number of Death Eaters, including both Severus Snape and Barty Crouch Jr. While Dumbledore vouches for Snape's integrity, Crouch Sr. is horrified at this revelation and disowns his maniacal son, sending him to Azkaban. Upon returning to the present time, Dumbledore tells Harry that he is searching his memories for a clue as to why extraordinary events have taken place at Hogwarts since the start of the tournament. In the Triwizard Tournament's third and final task, the competitors are placed inside a hedge maze; their challenge is to reach the Triwizard Cup. Krum, acting under the Imperius curse, incapacitates Fleur and attempts to do the same to Cedric. Harry stops Cedric from attacking Krum, and the two run for the cup. When Cedric is trapped by vines, Harry frees him and the two claim a draw and grab hold of the cup together. The cup, which is a Portkey, transports the two champions to a graveyard where Wormtail and Voldemort are waiting for Harry. Wormtail murders Cedric, traps Harry, then performs a ritual that rejuvenates Voldemort, who then summons the Death Eaters  and bids them to witness a duel between their Dark Lord and his nemesis. As Harry, who is tortured by Voldemort, fights him, a connection called Priori Incantatem occurs between their wands. Harry's wand forces Voldemort's to disgorge the spirits of the people Voldemort has most recently murdered, including Harry's parents, Frank Bryce and Cedric. Harry is briefly protected by the spirits and escapes with Cedric's body  using the cup. Upon his return, Harry tells Dumbledore and Minister for Magic Cornelius Fudge that Voldemort has returned and is responsible for Cedric's death. Mad-Eye leads a devastated Harry back to the castle, where his questions make Harry suspicious. Mad-Eye reveals it was he who put Harry's name in the Goblet, assisted Cedric and Neville in helping Harry, cursed Krum and so on. Dumbledore, Snape and McGonagall arrive and force Veritaserum, a truth-telling potion down Mad-Eye's throat. He reveals he is not Alastor 'Mad-Eye' Moody and the real one is imprisoned in a magical trunk minus his magical eye and fake leg. The false Mad-Eye's Polyjuice Potion  wears off and he is revealed as Barty Crouch Jr., who shows a pulsing Dark Mark tattoo on his forearm meaning Voldemort's returned. Soon after, students and staff of Hogwarts, Durmstrang and Beauxbatons gather in the Great Hall to say farewell to Cedric. Dumbledore exhorts them to stand together against Voldemort, as the representatives from Durmstrang and Beauxbatons leave Hogwarts."
#s1 = "Test pilot John Mitchell  disappoints his wife Mary  by refusing to increase their unsuccessful bid for a house. What she does not know is that the aircraft manufacturing company he works for is in desperate financial straits. Owner Reginald Conway  needs to get Ashmore  to place an order soon or the firm will go bankrupt. Mitchell takes the only prototype of a new airplane for a flight, with Ashmore and several others aboard. During testing, one engine catches on fire. Ashmore and the others safely parachute to safety. Then, despite Conway's order and the urgings of others, Mitchell decides to try to land the plane rather than crashing it into the sea. However, he has to fly back and forth for half an hour to use up the fuel. During the tense wait, co-worker Snowden  takes it upon herself to notify Mitchell's wife. Mary goes to the airfield and watches as her husband manages to land safely. Later, at home, she demands to know why he risked his life when everyone told him to bail out. He explains that it was his duty and the company's fate hung in the balance. Then he phones their real estate agent and agrees to the seller's price."
#s1 = "The story is set in Chongyang  and Sanming . Zhou Yu, a ceramics artist from Sanming falls in love with the poet Chen Qing, who lives in Chongyang, a town several hundred kilometers from Sanming. During the train trips between Sanming and Chongyang, she also meets Zhang Qiang, a veterinary surgeon. Gong Li plays two characters who only differ by their hair styles, namely Zhou Yu and the short-haired Xiu. The film is pieced together with many flashbacks in no particular chronological order. The relation between the two women is unclear until the end of the film."
#s1 = "When Voyager encounters a seemingly friendly alien ship its crew has never seen before, the Nasari, Ensign Kim instinctively fires Voyager's weapons upon it without orders. Harry claims that a tetryon surge being emitted by the ship was the aliens preparing to attack them. Voyager sustains heavy damage while battling the enemy but manages to make them withdraw. Kim is relieved of duty, pending investigation of his overtly hostile actions. Janeway later confirms that Harry's 'instincts' were correct, that the ship was indeed preparing to attack. Harry recounts other episodes of déjà vu since entering that region of space. Later that night he has a strange dream, and when he awakens, he discovers a strange reddish rash on his head. The Doctor cannot find an initial cause. When the Nasari return with more ships, Harry's instincts tell him to flee to a nearby planet where they are defended and then hailed by the Taresians. Captain's log, stardate 50732.4. The Taresians have escorted us back to their home world so we can continue to investigate their claim that Ensign Kim is a member of their race. Once on the planet, the Taresians attempt to explain Harry's strange behavior. According to them, Harry is Taresian and was sent to Earth as an embryo where he was implanted into his 'human mother.' He took on the genetic traits of his human parents, but was programmed with both genetic knowledge and an urge to one day return to Taresia. When he entered the Taresian region of space, his dormant Taresian genes began to reactivate, explaining his current behavior. Harry befriends another male Taresian and learns that the planet's population is 90% female, and that males are very rare and thus very valuable and prized. While Harry remains on the planet in an attempt to better understand his recent changes, Voyager leaves to attempt to make contact with the Nasari and negotiate safe passage from the system. Harry learns that because males are so rare on Taresia, he is valuable breeding stock. He attends the marriage ceremony of his new male friend who is bonded with three Taresian females. On Voyager, the doctor discovers something strange about Harry's DNA: it was implanted, possibly by a virus, and it is not natural to his body. Janeway also realizes that a defensive grid has been raised around Taresia preventing them from communicating with Harry or transporting to the surface. After spending the night on the planet, Harry suffers more strange dreams which cause doubts to begin creeping into his mind. Two very 'interested' females attempt to calm him, but he resists their attempts. Now outright suspicious, Harry goes to see his male friend. When he enters his quarters he discovers the desiccated remains of his friend. He is confronted by several females where they reveal the truth to him. He was not born on Taresia. Because natural born males are so rare, the Taresians devised an 'artificial' way to procreate their species. They implant Taresian genes into male members of other species which rewrites their DNA and makes it compatible with Taresian females. Even more horrifying is that the 'reproductive process' means draining the males of their DNA where it is then implanted into females, causing the death of the males. As a result, new males are constantly 'created' and harvested. Harry attempts to flee and is pursued by many females and their leader. Just when they are about to capture him, Voyager penetrates the grid and beams Harry on board. They flee the system with a Taresian ship in pursuit. The Nasari appear but after initially focusing their fire on Voyager, they then attack the Taresians, allowing Voyager to escape. The Doctor is later able to extract the foreign DNA and successfully returns Harry to his human form."
#extract_from_line(s1)


cluster_name = "plot_summaries.txt"
ff = open('output_plotsummaries.pkl')
list_dict = []
list_stryline = []
#all_dict = extract_keywords(filepath + "plot_summaries.txt")
with open (filepath + cluster_name, mode = 'r', encoding = "utf-8") as f:
    with open('output_strylines.pkl', 'wb') as fff:
        with open ('output_plotsummaries.pkl', 'wb') as ff:
            line = f.readline()
            while line:
                temp, flag = extract_from_line(line)
                if flag == True:
                    try:
                        for word in temp:
                            glove_model.most_similar(word)
                        list_dict.append(temp)
                        list_stryline.append(line)
                    except:
                        None
                line = f.readline()
            pickle.dump(list_dict, ff)
            ff.close()
        pickle.dump(list_stryline, fff)
        fff.close()
f.close()




#RAKE - keyword extractions