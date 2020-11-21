mkdir -p ~/.streamlit/                                               
echo "\                       
[server]                    
port = $PORT                      
enableCORS = false                       
headless = true                       
                    
" > ~/.streamlit/config.toml
