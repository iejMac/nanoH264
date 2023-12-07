def log_compression_report(report):
  for k, v in report.items():
    if k != "grades":
      print(f"{k}: {v}")
