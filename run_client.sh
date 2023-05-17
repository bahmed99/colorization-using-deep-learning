#!/bin/bash
cd ./src/view && npm install

# Resolve the issue of loading and pop-up during colorization.
config_file="node_modules/react-scripts/config/webpackDevServer.config.js"
if [ -f "$config_file" ] && grep -q "ignored: ignoredFiles(paths.appSrc)," "$config_file"; then
  sed -i 's|ignored: ignoredFiles(paths\.appSrc),|ignored: [ ignoredFiles(paths.appSrc), paths.appPublic ],|g' "$config_file"
  echo "Value of ignored replaced successfully in node modules."
else
  echo "webpackDevServer.config.js file not found or old string not found in file."
fi

# Start the client
npm start