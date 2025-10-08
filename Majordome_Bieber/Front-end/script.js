document.getElementById('send-command').addEventListener('click', () => {
    const command = document.getElementById('command-input').value;
    alert(`Commande envoyée à Justin Bieber : ${command}`);
});

// TODO : remplacer les alertes par appels API vers Roo
// Onglets Connexion / Inscription
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        tabContents.forEach(c => c.style.display = 'none');
        document.getElementById(btn.dataset.tab).style.display = 'block';
    });
});

document.getElementById('login-btn').addEventListener('click', () => {
    alert('Ouverture de la fenêtre de connexion...');
});

document.getElementById('signup-btn').addEventListener('click', () => {
    alert('Ouverture de la fenêtre de création de compte...');
});

