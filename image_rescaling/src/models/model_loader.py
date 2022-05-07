import torch
import time
import yaml

def load_config(cfg_filename):
  with open("./configs/" + cfg_filename, "r") as inp:
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

def save_network(inn, optimizer, epoch, min_training_loss, all_test_losses, all_test_psnr_y, name):
    save_filename = f"models/model_{int(time.time())}_{name}.pth"

    torch.save({
        "epoch": epoch,
        "model_state_dict":     inn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "min_training_loss":    min_training_loss,
        "min_test_loss":        min(all_test_losses)            if len(all_test_losses) > 0 else 99999,
        "max_test_psnr_y":      max(all_test_psnr_y)            if len(all_test_psnr_y) > 0 else 99999
    }, save_filename)

def load_network(inn, path, optimizer=None):
    print(f"Loading {path}")

    checkpoint = torch.load(path)
    inn.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return inn, optimizer, checkpoint.get("epoch", 0), checkpoint.get("min_training_loss", 99999), checkpoint.get("min_test_loss", 99999), checkpoint.get("max_test_psnr_y", -999)
