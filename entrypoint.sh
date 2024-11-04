mkdir -p /root/.config
echo 'Ensured /root/.config directory exists'
if id "ubuntu" &>/dev/null; then
    echo 'User ubuntu already exists'
fi
