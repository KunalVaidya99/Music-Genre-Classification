mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"170020003@iitdh.ac.in\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
