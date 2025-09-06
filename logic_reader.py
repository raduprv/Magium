from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import groupby
import itertools
import json
import pathlib
import pprint
import re
from typing import Any
import warnings
import sympy as sp

root_folder = pathlib.Path("./chapters")
pp = pprint.PrettyPrinter(2)

DEBUG_SCENE = "B3-Ch06c-Herbs"

STATS_VARIABLE_NAMES = [
    "v_agility",
    "v_aura_hardening",
    "v_magical_sense",
    "v_bluff",
    "v_combat_technique",
    "v_ancient_languages",
    "v_perception",
    "v_premonition",
    "v_hearing",
    "v_reflexes",
    "v_toughness",
    "v_strength",
    "v_available_points",
    "v_magical_knowledge",
    "v_magical_power",
    "v_max_stat",
]
STATS_VARIABLE_NAMES += [f"{v}_aux" for v in STATS_VARIABLE_NAMES]

class Lexer:
    def __init__(self,data):
        self.data = data
        self.pos = 0
        self.peek = data[0]

    def next(self):
        line = self.peek
        self.pos += 1
        self.peek = data[self.pos]
        return line

    def eof(self):
        return self.pos >= len(data) - 1


def is_condition(line):
    return line[0] in ["+","*"]

def transform_var_name(var_name):
    new_name = "v_"+var_name.lower().replace(" ","_")
    if new_name == "v_speed":
        return "v_agility"
    if new_name == "v_observation":
        return "v_perception"
    else:
        return new_name

def parse_condition(t: str):
    if t in [">","<",">=","<="]:
        return t
    elif t == "=":
        return "=="
    elif t == "<>":
        return "!="
    else:
        raise ValueError(t)

def groupby_unsorted(input, key=lambda x:x):
  yielded = set()
  keys = [ key(element) for element in input ]
  for i, wantedKey in enumerate(keys):
    if wantedKey not in yielded:
      yield (wantedKey,
          [input[j] for j in range(i, len(input)) if keys[j] == wantedKey])
    yielded.add(wantedKey)

@dataclass
class Event:
    type: str = ""
    conditions: dict[str,Any] = field(default_factory=lambda:{"variables":{}})
    results: dict[str,Any] = field(default_factory=lambda:{"add_buttons":[],"set_variables":{},"paragraphs":[]})



def compare(value,comparison):
    return eval(f"{value} {parse_condition(comparison.type)} {comparison.value}")

def is_compatible(comp1,comp2):
    if comp1 is None or comp2 is None:
        return False
    # Handling cases like toughness > 2 and toughness < 3
    if isinstance(comp1,sp.StrictGreaterThan) and isinstance(comp2,sp.StrictLessThan):
        if int(comp1.rhs) == int(comp2.rhs) - 1:
            return False
    if isinstance(comp1,sp.StrictLessThan) and isinstance(comp2,sp.StrictGreaterThan):
        if int(comp1.rhs) - 1 == int(comp2.rhs):
            return False
    return sp.simplify(sp.And(comp1,comp2)) != False

def all_is_compatible(comp_set1,comp_set2):
    return all(is_compatible(comp_set1[key],comp_set2.get(key)) for key in comp_set1)

def is_compatible_permissive(comp1,comp2):
    if comp1 is None or comp2 is None:
        return True
    if isinstance(comp1,sp.StrictGreaterThan) and isinstance(comp2,sp.StrictLessThan):
        if int(comp1.rhs) == int(comp2.rhs) - 1:
            return False
    if isinstance(comp1,sp.StrictLessThan) and isinstance(comp2,sp.StrictGreaterThan):
        if int(comp1.rhs) - 1 == int(comp2.rhs):
            return False
    return sp.simplify(sp.And(comp1,comp2)) != False

def all_is_compatible_permissive(comp_set1,comp_set2):
    return all(is_compatible_permissive(comp_set1[key],comp_set2.get(key)) for key in comp_set1)

def apply_condition_to_sympy(x,cond_type,cond_value):
    if cond_type == ">":
        return x > cond_value
    elif cond_type == "<":
        return x < cond_value
    elif cond_type == ">=":
        return x >= cond_value
    elif cond_type == "<=":
        return x <= cond_value
    elif cond_type == "=":
        return sp.Eq(x,cond_value)
    elif cond_type == "<>":
        return sp.Ne(x,cond_value)
    else:
        raise ValueError

def is_total_pattern(conditions: list):
    combined_condition = sp.Or(*conditions)
    simplified_condition = sp.simplify(combined_condition)
    return simplified_condition

@dataclass(frozen=True)
class AchievementEarned:
    text: str
    variable: str

@dataclass
class Paragraph:
    id: int
    version: int
    conditions: dict = field(default_factory=dict)

@dataclass
class ParagraphGroup:
    paragraphs: list[tuple[int,int]] = field(default_factory=list) # List of (id,version)
    conditions: dict = field(default_factory=dict)

@dataclass
class Response:
    text: str = field(default_factory=str)
    new_scene: str = field(default_factory=str)
    set_variables: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)
    achievements: list[AchievementEarned] = field(default_factory=list)
    special: str = ""

@dataclass
class SceneVariableSet:
    name: str
    value: str
    conditions: dict = field(default_factory=dict)

@dataclass
class StatsCheck:
    text: str
    successful: bool
    conditions: dict = field(default_factory=dict)

@dataclass
class Path:
    conditions: dict[str,Any]
    responses: list[Response]
    paragraphs: list[Paragraph]
    set_variables: list[SceneVariableSet]
    text_only: bool = False

    def __repr__(self) -> str:
        text = "Path\n"
        text += f"\tConditions: {self.conditions}\n"
        text += f"\tParagraphs: {self.paragraphs}\n"
        text += f"\tResponses: {self.responses}\n"
        text += f"\tSet variables: {self.set_variables}\n"
        text += f"\tText only: {self.text_only}\n"
        return text


@dataclass
class Scene:
    id: str
    paths: list[Path] = field(default_factory=list) 
    paragraphs: list[Paragraph] = field(default_factory=list) 
    responses: list[Response] = field(default_factory=list) 
    set_variables: list[SceneVariableSet] = field(default_factory=list)
    achievements: list[AchievementEarned] = field(default_factory=list)

    def to_json(self,paragraphs):
        text = ""
        for par in self.paragraphs:
            if par.id not in paragraphs[par.version]:
                print(f"Paragraph {par.id} not found in {par.version}")
                continue
            if par.conditions != {}:
                condition_string = ' && '.join([f"(locals.{var} || 0) {parse_condition(cond.type)} {cond.value}" for var,cond in par.conditions.items()])
                text += f"<% if({condition_string}) {{%>"
            text += paragraphs[par.version][par.id]
            if par.conditions != {}:
                text += "<% } %>" #}
        text = re.sub(r"(\n){2,}", r"\n\n", text)
        text = text.replace("\n","<br/>")
        val = {
            "id": self.id,
            "text": text,
            "responses": [
                {
                    "text": response.text,
                    "target": response.new_scene,
                    "set_variables": response.set_variables,
                    "conditions": {var: {"type":parse_condition(condition.type),"value":condition.value} for var,condition in response.conditions.items()},
                    "special": response.special,
                }
                for response in self.responses
            ],
            "set_variables": [
                {
                    "name": set_variable.name,
                    "value": set_variable.value,
                    "conditions": {var: {"type":parse_condition(condition.type),"value":condition.value} for var,condition in set_variable.conditions.items()},
                }
                for set_variable in self.set_variables
            ]
        }
        return val

    def to_magium(self,paragraphs,var_possible_values: dict[str,set]):
        text = f"ID: {self.id}\nTEXT:\n\n"

        set_variables = self.merge_set_variables()
        for set_variable in set_variables:
            text += f'set({set_variable.name},{set_variable.value})'
            if set_variable.conditions != True:
                text += f' if ({sp.jscode(set_variable.conditions)})'
            text += "\n"

        # This variable is missing from the original events for an unknown reason
        if self.id == "Ch6-Lose-fight":
            text += "set(v_maximized_stats_used,1)\n"


        for achievement in set(self.achievements):
            text += f'achievement("{achievement.text}",{achievement.variable})\n'

        paragraph_groups = self.merge_paragraphs(var_possible_values)
        for par_group in paragraph_groups:
            if par_group.conditions != True:
                text += f"#if({sp.jscode(par_group.conditions)}) {{\n"
            for par_tuple in par_group.paragraphs:
                if par_tuple[0] not in paragraphs[par_tuple[1]]:
                    print(f"Paragraph {par_tuple[0]} not found in {par_tuple[1]}")
                    continue
                text += paragraphs[par_tuple[1]][par_tuple[0]]
            if par_group.conditions != True:
                text += "}\n" #}
        text = re.sub(r"(\n){2,}", r"\n\n", text)

        responses = self.merge_responses(var_possible_values)
        for response in responses:
            # Filter values with aux to avoid a bug (https://github.com/thuiop/magium-dev/issues/89)
            set_variables_string = ", ".join([f"{name} = {value}" for name,value in response.set_variables.items() if "aux" not in str(value)])
            text += f'choice("{response.text}", {response.new_scene}, {set_variables_string}'
            if response.special != "":
                 text += f", special:{response.special}"
            text += ")"
            if response.conditions != True:
                text += f' if ({sp.jscode(response.conditions)})'
            text += "\n"

        return text

    def merge_paragraphs(self,var_possible_values: dict[str,set]):
        grouped_paragraphs = groupby_unsorted(self.paragraphs[::-1],key=lambda r:(r.id,r.version))
        new_paragraphs = []
        for (id,version),group in grouped_paragraphs:
            group = list(group)
            new_conditions = sp.simplify_logic(sp.Or(*[sp.And(*paragraph.conditions.values()) for paragraph in group]),form="dnf")
            new_paragraphs.append(Paragraph(id,version,new_conditions))

        # Necessary to ensure paragraphs remain in the correct order
        new_paragraphs = sorted(new_paragraphs[::-1],key=lambda x:x.version)

        # If we have several options for a paragraph version, ensure they are incompatible
        for i in range(3):
            paragraphs_i = [paragraph for paragraph in new_paragraphs if paragraph.version == i]
            if len(paragraphs_i):
                for j,paragraph in enumerate(paragraphs_i):
                    for other_paragraph in paragraphs_i[j+1:]:
                        if is_compatible(paragraph.conditions,other_paragraph.conditions):
                            paragraph.conditions = sp.simplify(sp.And(paragraph.conditions,sp.Not(other_paragraph.conditions)))
        
        cond_vars = set(itertools.chain.from_iterable([paragraph.conditions.keys() for paragraph in self.paragraphs]))
        cond_vars = {var for var in cond_vars if var not in STATS_VARIABLE_NAMES}
        for var in cond_vars:
            tautological_cond = sp.Or(*[sp.Eq(sp.symbols(var),int(possible_value)) for possible_value in var_possible_values[var]])
            for new_paragraph in new_paragraphs:
                try:
                    new_paragraph.conditions = sp.simplify_logic(new_paragraph.conditions,dontcare=~tautological_cond)
                except:
                    print(var,var_possible_values[var])
                    continue

        paragraph_groups = {}
        for paragraph in new_paragraphs:
            key = str(paragraph.conditions) 
            if key not in paragraph_groups:
                paragraph_groups[key] = ParagraphGroup(conditions=paragraph.conditions)
            paragraph_groups[key].paragraphs.append((paragraph.id,paragraph.version))

        return paragraph_groups.values()


    def merge_responses(self, var_possible_values: dict[str,set]):
        grouped_responses = groupby_unsorted(self.responses,key=lambda r:(r.text,str(r.set_variables),r.special))
        new_responses = []
        for test,group in grouped_responses:
            group = list(group)
            new_response = deepcopy(group[0])
            new_response.conditions = sp.simplify_logic(sp.Or(*[sp.And(*response.conditions.values()) for response in group]),form="cnf")

            initial_conditions = sp.simplify_logic(new_response.conditions,form="dnf")

            cond_vars = set(itertools.chain.from_iterable([response.conditions.keys() for response in group]))
            cond_vars = {var for var in cond_vars if var not in STATS_VARIABLE_NAMES}
            for var in cond_vars:
                tautological_cond = sp.Or(*[sp.Eq(sp.symbols(var),int(possible_value)) for possible_value in var_possible_values[var]])
                try:
                    #new_response.conditions = new_response.conditions.subs(tautological_cond,True)
                    new_response.conditions = sp.simplify_logic(new_response.conditions,dontcare=~tautological_cond)
                except:
                    print(var,var_possible_values[var])
                    continue
            # print("Simplified")
            # print(new_response.conditions)
            new_response.conditions = sp.simplify_logic(new_response.conditions,form="dnf")
            # print("Initial",initial_conditions)
            if len(str(initial_conditions)) < len(str(new_response.conditions)):
                new_response.conditions = initial_conditions
            # print(new_response.conditions)
            new_responses.append(new_response)
            
        # We can remove the conditions if all the responses have the same conditions
        if all([new_response.conditions == new_responses[0].conditions for new_response in new_responses]):
            for new_response in new_responses:
                new_response.conditions = True
        return new_responses


    def merge_set_variables(self):
        grouped_set_variables = groupby_unsorted(self.set_variables,key=lambda r:(r.name,r.value))
        new_set_variables = []
        for (name,value),group in grouped_set_variables:
            group = list(group)
            new_conditions = sp.simplify_logic(sp.Or(*[sp.And(*set_variable.conditions.values()) for set_variable in group]),form="dnf")
            new_set_variables.append(SceneVariableSet(name,value,new_conditions))
        return sorted(new_set_variables,key=lambda r:r.name)

class Parser:
    def __init__(self, lexer) -> None:
        self.lexer = lexer

    def parse_event(self):
        assert self.lexer.peek.startswith("*")
        event = Event()
        while len(current:=self.lexer.next()) > 1 and not self.lexer.eof():
            if is_condition(current):
                if match := re.search('current scene = "(?P<scene>.*)"',current):
                    event.conditions["scene"] = match.group("scene")
                elif match := re.search('current scene <> "(?P<scene>.*)"',current):
                    continue
                elif "Mouse pointer is over" in current:
                    event.type = "button"
                elif match := re.search('Button ID of Group.Buttons = (?P<button>.*)',current):
                    event.conditions["button_id"] = int(match.group("button"))-1
                elif match := re.search('(?P<variable>.*) (?P<comparison>=|<|>|<>|>=|<=) (?P<value>.*)',current[2:]):
                    if "counter" not in match.group("variable").lower():
                        var_name = transform_var_name(match.group("variable"))
                        # Do not use the aux variables in conditions
                        var_name = var_name.removesuffix("_aux")
                        new_condition = apply_condition_to_sympy(sp.symbols(var_name),match.group("comparison"),int(match.group("value")))
                        if var_name not in event.conditions["variables"]:
                            event.conditions["variables"][var_name] = new_condition
                        else:
                            event.conditions["variables"][var_name] = sp.And(new_condition,event.conditions["variables"][var_name])
            else:
                if match := re.search('List : Add line "(?P<button_name>.*)"',current):
                    event.type = "scene_load"
                    event.results["add_buttons"].append(match.group("button_name").replace('""','"'))
                elif match := re.search('Special : Set current scene to "(?P<scene>.*)"',current):
                    event.results["new_scene"] = match.group("scene")
                    event.results["set_variables"][transform_var_name("current scene")] = match.group("scene")
                elif match := re.search('Special : Set chapter save to 1',current):
                    event.results["special"] = "checkpoint_save"
                elif match := re.search('Special : Set chapter load to 1',current):
                    event.results["special"] = "checkpoint_load"
                elif match := re.search('Special : Set (?P<variable>.*) to (?P<value>.*)',current):
                    var_name = transform_var_name(match.group("variable"))
                    event.results["set_variables"][var_name] = match.group("value")
                elif match := re.search('Special : Add (?P<value>.*) to (?P<variable>.*)',current):
                    var_name = transform_var_name(match.group("variable"))
                    event.results["set_variables"][var_name] = "+"+match.group("value")
                elif match := re.search('Special : Subtract (?P<value>.*) from  (?P<variable>.*)',current):
                    var_name = transform_var_name(match.group("variable"))
                    event.results["set_variables"][var_name] = "-"+match.group("value")
                elif match := re.search('Scene text : Display paragraph (?P<paragraph>.*)',current):
                    event.type = "scene_load" if event.type == "" or event.type is None else event.type
                    event.results["paragraphs"].append((int(match.group("paragraph")),1))
                elif match := re.search('Scene text 2 : Display paragraph (?P<paragraph>.*)',current):
                    event.type = "scene_load" if event.type == "" or event.type is None else event.type
                    event.results["paragraphs"].append((int(match.group("paragraph")),2))
                elif match := re.search('scene 3 : Display paragraph (?P<paragraph>.*)',current):
                    event.type = "scene_load" if event.type == "" or event.type is None else event.type
                    event.results["paragraphs"].append((int(match.group("paragraph")),3))
                elif match := re.search('checks : Set alterable string to (?P<text>.*)',current):
                    if "Checkpoint reached" in match.group("text"):
                        pass
                        # event.type = "scene_load"
                elif match := re.search('storyboard controls : Jump to frame "Stats"',current):
                    event.results["special"] = "stats"
                elif match := re.search('storyboard controls : Jump to frame "Save load game"',current):
                    if event.results.get("special","") == "":
                        event.results["special"] = "saves"
                elif match := re.search('Achievement title : Set alterable string to "(?P<achievement>.*)"',current):
                    achievement_var = [var for var in event.conditions["variables"] if "_ac_" in var][0]
                    if event.type == "": event.type = "achievement"
                    event.results["achievement"] = AchievementEarned( match.group('achievement'),achievement_var)
                    event.conditions["variables"] = {var:condition for var, condition in event.conditions["variables"].items() if "_ac_" not in var}
                    

        # Paragraph from "Scene text 2" or "scene 3" should always be 2nd and third, but sometimes are out of order in the logic file
        event.results["paragraphs"] = sorted(event.results["paragraphs"],key=lambda x:x[1])

        # Sometimes occur
        if event.type == "" and len(event.results["set_variables"]) and "scene" in event.conditions:
            event.type = "extra_set"


        return event

    def parse(self):
        events = []
        while not self.lexer.eof():
            while not self.lexer.peek.startswith("*") and not self.lexer.eof():
                self.lexer.next()
            events.append(self.parse_event())
        return events
            

chapters = (
    [f"ch{num}" for num in list(range(1,11))+["11a","11b"]]
    + [f"b2ch{num}" for num in [1,2,3,"4a","4b","5a","5b",6,7,8,"9a","9b","10a","10b","11a","11b","11c"]]
    + [f"b3ch{num}" for num in [1,"2a","2b","2c","3a","3b","4a","4b","5a","5b","6a","6b","6c","7a","8a","8b","9a","9b","9c","10a","10b","10c","11a","12a","12b"]]
)
# chapters = ["b3ch6c"]
verbose = False
var_possible_values = defaultdict(set)
for chapter in chapters:
    filename = root_folder/chapter/"logic.txt"

    events = []
    current_event = {}
    with open(filename,"r") as f:
        data = list(f.readlines())
    lexer = Lexer(data)
    parser = Parser(lexer)

    events = parser.parse()
    scenes = {}

    for event in [e for e in events if e.type == "scene_load"]:
        if verbose:
            print(event)
        if "scene" not in event.conditions:
            print("No scene specified...")
            continue
        scene_id = event.conditions["scene"]
        if scene_id not in scenes:
            scenes[scene_id] = Scene(scene_id)

        scenes[scene_id].paths.append(Path(
            conditions=event.conditions["variables"],
            responses=[Response(button) for button in event.results["add_buttons"]],
            paragraphs=[Paragraph(paragraph[0],paragraph[1]) for paragraph in event.results["paragraphs"]],
            set_variables=[SceneVariableSet(variable,int(value)) for variable, value in event.results["set_variables"].items()],
        ))

    # Handling some more complex scenes
    for scene in scenes.values():
        for i, path in enumerate(scene.paths):
            path_conditions = {key:val for key,val in path.conditions.items()}
            for set_variable in path.set_variables:
                path_conditions[set_variable.name] = sp.Eq(sp.symbols(set_variable.name),set_variable.value)

            should_be_neutralized = False
            for other_path in scene.paths[i+1:]:
                # If the path does not have responses, it cannot override
                if len(other_path.responses) == 0:
                    continue

                other_path_conditions = {key:val for key,val in other_path.conditions.items()}
                # for set_variable in other_path.set_variables:
                #     other_path_conditions[set_variable.name] = sp.Eq(sp.symbols(set_variable.name),set_variable.value)

                is_set_in_any_path = False
                for condition in other_path_conditions:
                    is_set_in_any_path |= any([condition in [x.name for x in path.set_variables] for path in scenes[scene_id].paths])

                # If the path is compatible, it may overwrite the responses
                if all_is_compatible_permissive(path_conditions,other_path_conditions) and not is_set_in_any_path:
                    should_be_neutralized = sp.Or(should_be_neutralized,sp.And(*other_path_conditions.values()))

            for condition in path_conditions.values():
                if not isinstance(should_be_neutralized,bool):
                    should_be_neutralized = should_be_neutralized.subs(condition,True)
            
            # If there is a combination of overwriting paths that covers everything, neutralize it
            if sp.simplify(should_be_neutralized) == True:
                path.text_only = True
                path.responses = []

    for event in [e for e in events if e.type == "button"]:
        if verbose:
            print("#"*60)
            print(event)
        if "scene" not in event.conditions:
            print("No scene specified...")
            continue
        scene_id = event.conditions["scene"]
        if scene_id not in scenes:
            raise ValueError(f"The scene '{scene_id} was not found ! List of available scenes : {' '.join(scenes.keys())}'")


        set_variables = []
        for path in scenes[scene_id].paths:
            set_variables += [(set_variable,path.conditions) for set_variable in path.set_variables]

        grouped_set_variables = list(groupby_unsorted(set_variables,key=lambda r:(r[0].name)))
        grouped_set_variables = [(group[0],[condition[1] for condition in group[1]]) for group in grouped_set_variables]

        for set_variable, conditions_list in grouped_set_variables:
            for path in scenes[scene_id].paths:
                if path.text_only:
                    continue

                if (
                    set_variable not in path.conditions
                    and set_variable not in [v.name for v in path.set_variables]
                    and not any([all_is_compatible(path.conditions,conditions) for conditions in conditions_list])
                    and set_variable not in STATS_VARIABLE_NAMES
                ):
                    path.set_variables.append(SceneVariableSet(set_variable,0))

        event_conditions = event.conditions["variables"]
        for path in scenes[scene_id].paths:
            if verbose:
                print(path)
                print("Event conditions:",event_conditions)
                print("Button id:",event.conditions["button_id"])
            fake_scene_conditions = (
                path.conditions
                | {scene_set_variable.name: apply_condition_to_sympy(sp.symbols(scene_set_variable.name),"=",scene_set_variable.value) for scene_set_variable in path.set_variables}
                #| {scene_set_variable_name: Comparison("=",0) for scene_set_variable_name in set_variable_names if scene_set_variable_name not in [v.name for v in path.set_variables]}
            )

            unrelated_conditions = {}
            for var,condition in event_conditions.items():
                if var not in fake_scene_conditions:
                    unrelated_conditions[var] = condition
            for var,condition in unrelated_conditions.items():
                fake_scene_conditions[var] = condition
            if verbose:
                if len(unrelated_conditions):
                    print("There are unrelated conditions:",unrelated_conditions)


            if not all_is_compatible(event_conditions,fake_scene_conditions) or path.responses==[]:
                if verbose:
                    print("Not compatible")
                continue


            if event.conditions["button_id"] >= len(path.responses):
                warnings.warn(f"Button id {event.conditions['button_id']} was not found !")
                continue
            response = path.responses[event.conditions["button_id"]]

            if len(unrelated_conditions):
                new_path = Path({},[],[],[])
                new_path.conditions = deepcopy(path.conditions) | event_conditions
                response = deepcopy(response)
                new_path.responses.append(response)
                scenes[scene_id].paths.append(new_path)
                if verbose:
                    print("Creating a fake path...")
                    print(new_path)

            if verbose:
                print("Chosen response:",response)

            if "new_scene" in event.results: 
                response.new_scene = event.results["new_scene"]

            for var,value in event.results["set_variables"].items():
                response.set_variables[var] = value 
            if "achievement" in event.results:
                achievement: AchievementEarned = event.results["achievement"]
                response.set_variables[achievement.variable] = 1
                # We do not know yet the following scene, so we store the achievements in the response for now
                response.achievements.append(achievement)
                
            if scene_id == "Ch11b-Ending":
                response.set_variables["v_first_book_purchased"] = "1"
            if scene_id == "B2-Ch11c-Ending":
                response.set_variables["v_second_book_purchased"] = "1"

            if "special" in event.results and response.special == "":
                response.special = event.results["special"]
            elif response.text == "Restart game":
                response.special = "restart" 
                response.set_variables = {k:val for k,val in response.set_variables.items() if k == "v_current_scene"}

            if verbose:
                print("New response:",response)
                print()

    # Only achievements earned at the beginning of a scene
    for event in [e for e in events if e.type == "achievement"]:
        if verbose:
            print(event)
        if "scene" not in event.conditions:
            print("No scene specified...")
            continue
        scene_id = event.conditions["scene"]
        if scene_id not in scenes:
            raise ValueError(f"The scene '{scene_id} was not found ! List of available scenes : {' '.join(scenes.keys())}'")

        scenes[scene_id].set_variables += [
            SceneVariableSet(set_variable_name,set_variable_value,event.conditions["variables"])
            for set_variable_name, set_variable_value in event.results["set_variables"].items()
        ]
        scenes[scene_id].achievements.append(event.results["achievement"])

    for scene in scenes.values():
        for path in scene.paths:
            for element_type in ["paragraphs","responses","set_variables"]:
                for element in getattr(path,element_type):
                    element.conditions = path.conditions
                    getattr(scene,element_type).append(element)
            scene.responses = [response for response in scene.responses if response.new_scene != "" or response.special != ""]
        
        for response in scene.responses:
            for achievement in response.achievements:
                scenes[response.new_scene].achievements.append(achievement)


    paragraphs = {1:{},2:{},3:{}}

    for i in [1,2,3]:
        print(chapter,i)
        print(root_folder/chapter/f"paragraphs{i}.txt")
        try:
            with open(root_folder/chapter/f"paragraphs{i}.txt","r") as f:
                data = f.readlines()
        except Exception:
            with open(root_folder/chapter/f"paragraphs{i}.txt","r",encoding="utf16") as f:
                data = f.readlines()

        current_par_index = 0
        current_par = ""
        for line in data:
            if match := re.match("\[(?P<par_num>[0-9]*)\](?P<text>.*)",line):
                paragraphs[i][current_par_index] = current_par
                current_par = match.group("text")+"\n"
                current_par_index = int(match.group("par_num"))
            else:
                current_par += line
        paragraphs[i][current_par_index] = current_par


    for event in [e for e in events if e.type == "extra_set"]:
        scenes[event.conditions["scene"]].set_variables += [SceneVariableSet(set_variable_name,set_variable_value,event.conditions["variables"]) for set_variable_name, set_variable_value in event.results["set_variables"].items()]

    # For simplifying tautological conditions
    for scene in scenes.values():
        for response in scene.responses:
            for set_variable_name,value in response.set_variables.items():
                if (isinstance(value,str) and value.isnumeric()) or isinstance(value,int):
                    var_possible_values[set_variable_name].add(int(value))
            for set_variable_name,value in response.conditions.items():
                if isinstance(value,sp.Eq):
                    var_possible_values[set_variable_name].add(value.rhs)
            
        for set_variable in scene.set_variables:
            if (isinstance(set_variable.value,str) and set_variable.value.isnumeric()) or isinstance(set_variable.value,int):
                var_possible_values[set_variable.name].add(int(set_variable.value))

    magium_vals = "\n\n".join(scene.to_magium(paragraphs, var_possible_values) for scene in scenes.values())
    with open(root_folder/f"{chapter}.magium","w") as f:
        f.write(magium_vals) 

@dataclass
class Achievement:
    title: str
    caption: str
    chapter: str
    variable: str = ""

class AchievementParser:
    def __init__(self, lexer) -> None:
        self.lexer = lexer
        self.achievements = defaultdict(list)

    def parse_event(self):
        assert self.lexer.peek.startswith("*")
        current_achievement = {}
        current_chapter = ""
        current_achievement_id = 0
        current_achievement_var = ""
        while len(current:=self.lexer.next()) > 1 and not self.lexer.eof():
            if is_condition(current):
                if match := re.search('achievement chapter = (?P<chapter>.*)',current):
                    current_chapter = match.group("chapter")
                elif match := re.search('Button ID of Button = (?P<button_id>.*)',current):
                    current_achievement_id = int(match.group("button_id")) - 1
                elif match := re.search('(?P<achievement_id>AC .*) = 1',current):
                    current_achievement_var = transform_var_name(match.group("achievement_id"))
            else:
                if match := re.search('List1 : Add line "(?P<achievement_title>.*)"',current):
                    current_achievement["title"] = match.group("achievement_title")
                elif match := re.search('List2 : Add line "(?P<achievement_caption>.*)"',current):
                    current_achievement["caption"] = match.group("achievement_caption")
                    self.achievements[current_chapter].append(Achievement(current_achievement["title"],current_achievement["caption"],current_chapter))
                elif match := re.search('Active 2 : Make invisible',current):
                    self.achievements[current_chapter][current_achievement_id].variable = current_achievement_var

    def parse(self):
        while True:
            while not self.lexer.peek.startswith("*") and not self.lexer.eof():
                self.lexer.next()
            if self.lexer.eof():
                break
            self.parse_event()
        return self.achievements

def chapter_name(id,book):
    return f"b{book}ch{id}"

for i in range(1,4):
    with open(f"chapters/achievements_logic{i}.txt","r") as f:
        data = f.readlines()

    lexer = Lexer(data)
    parser = AchievementParser(lexer)
    achievements = parser.parse()
    achievements_list = []
    for achievement_group in achievements.values():
        achievements_list += achievement_group
    achievements_json = [
        {
            "title": achievement.title,
            "caption": achievement.caption,
            "chapter": chapter_name(achievement.chapter,i),
            "variable": achievement.variable
        }
        for achievement in achievements_list
    ]

    achievements_json = {k: list(v) for k, v in groupby(achievements_json,key=lambda a:a["chapter"])}
    with open(root_folder/f"achievements{i}.json","w") as f:
        json.dump(achievements_json,f,indent=4)
