
cls_dict = {
    "mnist": {
        0: '0',
        1: '1', 
        2: '2', 
        3: '3', 
        4: '4', 
        5: '5', 
        6: '6', 
        7: '7', 
        8: '8', 
        9: '9',
    }, 
    
    "cifar10": {
        0: 'airplane',
        1: 'automobile', 
        2: 'bird', 
        3: 'cat', 
        4: 'deer', 
        5: 'dog', 
        6: 'frog', 
        7: 'horse', 
        8: 'ship', 
        9: 'truck',

    }, 
}


cls_ls = {
    "mnist": ['0', '1', '2', '3', '4', '5', '6', '7', '8',  '9'],


    "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 
    "cifar100": [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
        'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 
        'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 
        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 
        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 
        'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
        'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 
        'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ], 
    "cifar100-coarse": ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 'medium-sized_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'],
    

    "gtsrb": [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
        'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
        'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
        'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
        'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ], 


    "timgnet": [
        'Egyptian cat', 'reel', 'volleyball', 'rocking chair', 'lemon', 'bullfrog', 'basketball', 'cliff', 'espresso', 'plunger', 
        'parking meter', 'German shepherd', 'dining table', 'monarch', 'brown bear', 'school bus', 'pizza', 'guinea pig', 'umbrella', 
        'organ', 'oboe', 'maypole', 'goldfish', 'potpie', 'hourglass', 'seashore', 'computer keyboard', 'Arabian camel', 'ice cream', 
        'nail', 'space heater', 'cardigan', 'baboon', 'snail', 'coral reef', 'albatross', 'spider web', 'sea cucumber', 'backpack', 
        'Labrador retriever', 'pretzel', 'king penguin', 'sulphur butterfly', 'tarantula', 'lesser panda', 'pop bottle', 'banana', 
        'sock', 'cockroach', 'projectile', 'beer bottle', 'mantis', 'freight car', 'guacamole', 'remote control', 
        'European fire salamander', 'lakeside', 'chimpanzee', 'pay-phone', 'fur coat', 'alp', 'lampshade', 'torch', 'abacus', 
        'moving van', 'barrel', 'tabby', 'goose', 'koala', 'bullet train', 'CD player', 'teapot', 'birdhouse', 'gazelle', 
        'academic gown', 'tractor', 'ladybug', 'miniskirt', 'golden retriever', 'triumphal arch', 'cannon', 'neck brace', 'sombrero', 
        'gasmask', 'candle', 'desk', 'frying pan', 'bee', 'dam', 'spiny lobster', 'police van', 'iPod', 'punching bag', 'beacon', 
        'jellyfish', 'wok', "potter's wheel", 'sandal', 'pill bottle', 'butcher shop', 'slug', 'hog', 'cougar', 'crane', 'vestment', 
        'dragonfly', 'cash machine', 'mushroom', 'jinrikisha', 'water tower', 'chest', 'snorkel', 'sunglasses', 'fly', 'limousine', 
        'black stork', 'dugong', 'sports car', 'water jug', 'suspension bridge', 'ox', 'ice lolly', 'turnstile', 'Christmas stocking', 
        'broom', 'scorpion', 'wooden spoon', 'picket fence', 'rugby ball', 'sewing machine', 'steel arch bridge', 'Persian cat', 
        'refrigerator', 'barn', 'apron', 'Yorkshire terrier', 'swimming trunks', 'stopwatch', 'lawn mower', 'thatch', 'fountain', 
        'black widow', 'bikini', 'plate', 'teddy', 'barbershop', 'confectionery', 'beach wagon', 'scoreboard', 'orange', 'flagpole', 
        'American lobster', 'trolleybus', 'drumstick', 'dumbbell', 'brass', 'bow tie', 'convertible', 'bighorn', 'orangutan', 
        'American alligator', 'centipede', 'syringe', 'go-kart', 'brain coral', 'sea slug', 'cliff dwelling', 'mashed potato', 
        'viaduct', 'military uniform', 'pomegranate', 'chain', 'kimono', 'comic book', 'trilobite', 'bison', 'pole', 'boa constrictor', 
        'poncho', 'bathtub', 'grasshopper', 'walking stick', 'Chihuahua', 'tailed frog', 'lion', 'altar', 'obelisk', 'beaker', 
        'bell pepper', 'bannister', 'bucket', 'magnetic compass', 'meat loaf', 'gondola', 'standard poodle', 'acorn', 'lifeboat', 
        'binoculars', 'cauliflower', 'African elephant'
    ],     
    "timgnet-id": ['n02124075', 'n04067472', 'n04540053', 'n04099969', 'n07749582', 'n01641577', 'n02802426', 'n09246464', 'n07920052', 'n03970156', 'n03891332', 'n02106662', 'n03201208', 'n02279972', 'n02132136', 'n04146614', 'n07873807', 'n02364673', 'n04507155', 'n03854065', 'n03838899', 'n03733131', 'n01443537', 'n07875152', 'n03544143', 'n09428293', 'n03085013', 'n02437312', 'n07614500', 'n03804744', 'n04265275', 'n02963159', 'n02486410', 'n01944390', 'n09256479', 'n02058221', 'n04275548', 'n02321529', 'n02769748', 'n02099712', 'n07695742', 'n02056570', 'n02281406', 'n01774750', 'n02509815', 'n03983396', 'n07753592', 'n04254777', 'n02233338', 'n04008634', 'n02823428', 'n02236044', 'n03393912', 'n07583066', 'n04074963', 'n01629819', 'n09332890', 'n02481823', 'n03902125', 'n03404251', 'n09193705', 'n03637318', 'n04456115', 'n02666196', 'n03796401', 'n02795169', 'n02123045', 'n01855672', 'n01882714', 'n02917067', 'n02988304', 'n04398044', 'n02843684', 'n02423022', 'n02669723', 'n04465501', 'n02165456', 'n03770439', 'n02099601', 'n04486054', 'n02950826', 'n03814639', 'n04259630', 'n03424325', 'n02948072', 'n03179701', 'n03400231', 'n02206856', 'n03160309', 'n01984695', 'n03977966', 'n03584254', 'n04023962', 'n02814860', 'n01910747', 'n04596742', 'n03992509', 'n04133789', 'n03937543', 'n02927161', 'n01945685', 'n02395406', 'n02125311', 'n03126707', 'n04532106', 'n02268443', 'n02977058', 'n07734744', 'n03599486', 'n04562935', 'n03014705', 'n04251144', 'n04356056', 'n02190166', 'n03670208', 'n02002724', 'n02074367', 'n04285008', 'n04560804', 'n04366367', 'n02403003', 'n07615774', 'n04501370', 'n03026506', 'n02906734', 'n01770393', 'n04597913', 'n03930313', 'n04118538', 'n04179913', 'n04311004', 'n02123394', 'n04070727', 'n02793495', 'n02730930', 'n02094433', 'n04371430', 'n04328186', 'n03649909', 'n04417672', 'n03388043', 'n01774384', 'n02837789', 'n07579787', 'n04399382', 'n02791270', 'n03089624', 'n02814533', 'n04149813', 'n07747607', 'n03355925', 'n01983481', 'n04487081', 'n03250847', 'n03255030', 'n02892201', 'n02883205', 'n03100240', 'n02415577', 'n02480495', 'n01698640', 'n01784675', 'n04376876', 'n03444034', 'n01917289', 'n01950731', 'n03042490', 'n07711569', 'n04532670', 'n03763968', 'n07768694', 'n02999410', 'n03617480', 'n06596364', 'n01768244', 'n02410509', 'n03976657', 'n01742172', 'n03980874', 'n02808440', 'n02226429', 'n02231487', 'n02085620', 'n01644900', 'n02129165', 'n02699494', 'n03837869', 'n02815834', 'n07720875', 'n02788148', 'n02909870', 'n03706229', 'n07871810', 'n03447447', 'n02113799', 'n12267677', 'n03662601', 'n02841315', 'n07715103', 'n02504458'],
    "timgnet-full": ['Egyptian cat', 'reel', 'volleyball', 'rocking chair, rocker', 'lemon', 'bullfrog, Rana catesbeiana', 'basketball', 'cliff, drop, drop-off', 'espresso', "plunger, plumber's helper", 'parking meter', 'German shepherd, German shepherd dog, German police dog, alsatian', 'dining table, board', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'brown bear, bruin, Ursus arctos', 'school bus', 'pizza, pizza pie', 'guinea pig, Cavia cobaya', 'umbrella', 'organ, pipe organ', 'oboe, hautboy, hautbois', 'maypole', 'goldfish, Carassius auratus', 'potpie', 'hourglass', 'seashore, coast, seacoast, sea-coast', 'computer keyboard, keypad', 'Arabian camel, dromedary, Camelus dromedarius', 'ice cream, icecream', 'nail', 'space heater', 'cardigan', 'baboon', 'snail', 'coral reef', 'albatross, mollymawk', "spider web, spider's web", 'sea cucumber, holothurian', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'Labrador retriever', 'pretzel', 'king penguin, Aptenodytes patagonica', 'sulphur butterfly, sulfur butterfly', 'tarantula', 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'pop bottle, soda bottle', 'banana', 'sock', 'cockroach, roach', 'projectile, missile', 'beer bottle', 'mantis, mantid', 'freight car', 'guacamole', 'remote control, remote', 'European fire salamander, Salamandra salamandra', 'lakeside, lakeshore', 'chimpanzee, chimp, Pan troglodytes', 'pay-phone, pay-station', 'fur coat', 'alp', 'lampshade, lamp shade', 'torch', 'abacus', 'moving van', 'barrel, cask', 'tabby, tabby cat', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'bullet train, bullet', 'CD player', 'teapot', 'birdhouse', 'gazelle', "academic gown, academic robe, judge's robe", 'tractor', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'miniskirt, mini', 'golden retriever', 'triumphal arch', 'cannon', 'neck brace', 'sombrero', 'gasmask, respirator, gas helmet', 'candle, taper, wax light', 'desk', 'frying pan, frypan, skillet', 'bee', 'dam, dike, dyke', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'iPod', 'punching bag, punch bag, punching ball, punchball', 'beacon, lighthouse, beacon light, pharos', 'jellyfish', 'wok', "potter's wheel", 'sandal', 'pill bottle', 'butcher shop, meat market', 'slug', 'hog, pig, grunter, squealer, Sus scrofa', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'crane', 'vestment', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'mushroom', 'jinrikisha, ricksha, rickshaw', 'water tower', 'chest', 'snorkel', 'sunglasses, dark glasses, shades', 'fly', 'limousine, limo', 'black stork, Ciconia nigra', 'dugong, Dugong dugon', 'sports car, sport car', 'water jug', 'suspension bridge', 'ox', 'ice lolly, lolly, lollipop, popsicle', 'turnstile', 'Christmas stocking', 'broom', 'scorpion', 'wooden spoon', 'picket fence, paling', 'rugby ball', 'sewing machine', 'steel arch bridge', 'Persian cat', 'refrigerator, icebox', 'barn', 'apron', 'Yorkshire terrier', 'swimming trunks, bathing trunks', 'stopwatch, stop watch', 'lawn mower, mower', 'thatch, thatched roof', 'fountain', 'black widow, Latrodectus mactans', 'bikini, two-piece', 'plate', 'teddy, teddy bear', 'barbershop', 'confectionery, confectionary, candy store', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'scoreboard', 'orange', 'flagpole, flagstaff', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'trolleybus, trolley coach, trackless trolley', 'drumstick', 'dumbbell', 'brass, memorial tablet, plaque', 'bow tie, bow-tie, bowtie', 'convertible', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'orangutan, orang, orangutang, Pongo pygmaeus', 'American alligator, Alligator mississipiensis', 'centipede', 'syringe', 'go-kart', 'brain coral', 'sea slug, nudibranch', 'cliff dwelling', 'mashed potato', 'viaduct', 'military uniform', 'pomegranate', 'chain', 'kimono', 'comic book', 'trilobite', 'bison', 'pole', 'boa constrictor, Constrictor constrictor', 'poncho', 'bathtub, bathing tub, bath, tub', 'grasshopper, hopper', 'walking stick, walkingstick, stick insect', 'Chihuahua', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'lion, king of beasts, Panthera leo', 'altar', 'obelisk', 'beaker', 'bell pepper', 'bannister, banister, balustrade, balusters, handrail', 'bucket, pail', 'magnetic compass', 'meat loaf, meatloaf', 'gondola', 'standard poodle', 'acorn', 'lifeboat', 'binoculars, field glasses, opera glasses', 'cauliflower', 'African elephant, Loxodonta africana'],


}


# class_dict = {i: class_names[i] for i in range(100)}
# print(class_dict)


'''
    utils
'''
# cifar100:  fine - coarse
cifar100_coarse2fine = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'], 
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'], 
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'], 
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'], 
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'], 
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'], 
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'], 
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'], 
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'], 
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'], 
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'], 
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'], 
    'medium-sized_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'], 
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'], 
    'people': ['baby', 'boy', 'girl', 'man', 'woman'], 
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'], 
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'], 
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'], 
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'], 
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}
cifar100_fine2coarse = {'apple': 'fruit_and_vegetables', 'aquarium_fish': 'fish', 'baby': 'people', 'bear': 'large_carnivores', 'beaver': 'aquatic_mammals', 'bed': 'household_furniture', 'bee': 'insects', 'beetle': 'insects', 'bicycle': 'vehicles_1', 'bottle': 'food_containers', 'bowl': 'food_containers', 'boy': 'people', 'bridge': 'large_man-made_outdoor_things', 'bus': 'vehicles_1', 'butterfly': 'insects', 'camel': 'large_omnivores_and_herbivores', 'can': 'food_containers', 'castle': 'large_man-made_outdoor_things', 'caterpillar': 'insects', 'cattle': 'large_omnivores_and_herbivores', 'chair': 'household_furniture', 'chimpanzee': 'large_omnivores_and_herbivores', 'clock': 'household_electrical_devices', 'cloud': 'large_natural_outdoor_scenes', 'cockroach': 'insects', 'couch': 'household_furniture', 'crab': 'non-insect_invertebrates', 'crocodile': 'reptiles', 'cup': 'food_containers', 'dinosaur': 'reptiles', 'dolphin': 'aquatic_mammals', 'elephant': 'large_omnivores_and_herbivores', 'flatfish': 'fish', 'forest': 'large_natural_outdoor_scenes', 'fox': 'medium-sized_mammals', 'girl': 'people', 'hamster': 'small_mammals', 'house': 'large_man-made_outdoor_things', 'kangaroo': 'large_omnivores_and_herbivores', 'keyboard': 'household_electrical_devices', 'lamp': 'household_electrical_devices', 'lawn_mower': 'vehicles_2', 'leopard': 'large_carnivores', 'lion': 'large_carnivores', 'lizard': 'reptiles', 'lobster': 'non-insect_invertebrates', 'man': 'people', 'maple_tree': 'trees', 'motorcycle': 'vehicles_1', 'mountain': 'large_natural_outdoor_scenes', 'mouse': 'small_mammals', 'mushroom': 'fruit_and_vegetables', 'oak_tree': 'trees', 'orange': 'fruit_and_vegetables', 'orchid': 'flowers', 'otter': 'aquatic_mammals', 'palm_tree': 'trees', 'pear': 'fruit_and_vegetables', 'pickup_truck': 'vehicles_1', 'pine_tree': 'trees', 'plain': 'large_natural_outdoor_scenes', 'plate': 'food_containers', 'poppy': 'flowers', 'porcupine': 'medium-sized_mammals', 'possum': 'medium-sized_mammals', 'rabbit': 'small_mammals', 'raccoon': 'medium-sized_mammals', 'ray': 'fish', 'road': 'large_man-made_outdoor_things', 'rocket': 'vehicles_2', 'rose': 'flowers', 'sea': 'large_natural_outdoor_scenes', 'seal': 'aquatic_mammals', 'shark': 'fish', 'shrew': 'small_mammals', 'skunk': 'medium-sized_mammals', 'skyscraper': 'large_man-made_outdoor_things', 'snail': 'non-insect_invertebrates', 'snake': 'reptiles', 'spider': 'non-insect_invertebrates', 'squirrel': 'small_mammals', 'streetcar': 'vehicles_2', 'sunflower': 'flowers', 'sweet_pepper': 'fruit_and_vegetables', 'table': 'household_furniture', 'tank': 'vehicles_2', 'telephone': 'household_electrical_devices', 'television': 'household_electrical_devices', 'tiger': 'large_carnivores', 'tractor': 'vehicles_2', 'train': 'vehicles_1', 'trout': 'fish', 'tulip': 'flowers', 'turtle': 'reptiles', 'wardrobe': 'household_furniture', 'whale': 'aquatic_mammals', 'willow_tree': 'trees', 'wolf': 'large_carnivores', 'woman': 'people', 'worm': 'non-insect_invertebrates'}

