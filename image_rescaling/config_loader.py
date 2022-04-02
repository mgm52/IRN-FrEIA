import yaml

def load_config(path):
  with open(path, "r") as inp:
    try:
      config = yaml.safe_load(inp)
      print(f"Successfully loaded config:\n{config}")
      return config
    except yaml.YAMLError as ex:
      # Todo: consider handling this exeption differently
      raise ex

def check_keys(d, keys):
  for k in keys:
    if k not in d:
      raise Exception(f"Config file is missing key '{k}'. It was expected to contain keys:\n {keys}, but instead only contains keys:\n {list(d.keys())}.")
  return True