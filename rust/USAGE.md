# Utiliser `kyutai-pocket-tts` comme bibliothèque

Texte en entrée, PCM en sortie, dans votre processus. Pas de sous-processus, pas de serveur, pas de
Python.

## Déclarer la dépendance

Le paquet cargo s'appelle **`kyutai-pocket-tts`** ; la bibliothèque qu'il contient s'appelle
**`pocket_tts`**, et c'est ce qu'on écrit dans les `use`.

Les deux diffèrent parce que `pocket-tts` sur crates.io appartient à un crate tiers sans rapport,
que crates.io normalise `-` et `_` — donc `pocket_tts` désigne ce même crate — et qu'il n'offre
aucun namespace pour les distinguer.

```toml
[dependencies]
kyutai-pocket-tts = { version = "0.1", default-features = false }

# Ou, pour garder la clé `pocket-tts` dans votre manifeste :
pocket-tts = { package = "kyutai-pocket-tts", version = "0.1", default-features = false }

# Ou depuis un checkout, au choix :
kyutai-pocket-tts = { path = "../../../fork-pocket-tts/rust", default-features = false }
```

`default-features = false` coupe la feature `cli`, c'est-à-dire la commande, le serveur web, et avec
eux axum, clap, tokio, tokio-stream, tower-http et tempfile. Une bibliothèque qui synthétise trois
mots n'a aucune raison de compiler une pile HTTP.

### Pour ARIA

`crates/voice/Cargo.toml` déclare `pocket-tts = { path = … }`, qui **ne résout plus** depuis le
renommage. Une ligne à corriger, et aucun `use` à toucher :

```toml
pocket-tts = { package = "kyutai-pocket-tts", path = "../../../fork-pocket-tts/rust", default-features = false }
```

ARIA y gagne par ailleurs quelque chose. Son `Cargo.lock` portait **deux `ptts`** : `0.4.0-rc.3`
tiré du fork git par ce crate, et `0.5.0` de crates.io qu'ARIA déclare en direct. Deux moteurs
d'inférence, deux `xn`, dans le même binaire. Ce crate étant passé sur `ptts = "0.5"`, le graphe se
réduit à un seul.

## Le minimum qui parle

```rust
use pocket_tts::audio::StreamingWavWriter;
use pocket_tts::config::Config;
use pocket_tts::model::{LoadOptions, TtsModel};
use pocket_tts::{F32, defaults};
use std::sync::atomic::AtomicBool;

let (config, origin) = Config::load(Some("english"), None)?;
let model: TtsModel<F32> = TtsModel::load(config, origin, &LoadOptions::default())?;

let voice = model.get_state_for_audio_prompt("alba", false)?;

let mut pcm: Vec<f32> = Vec::new();
model.generate_audio_stream(
    &voice,
    "Bonjour, ceci sort d'une bibliothèque.",
    defaults::MAX_TOKEN_PER_CHUNK,
    None,                       // frames_after_eos : la valeur du modèle
    &AtomicBool::new(false),    // mettre à true coupe la génération
    &mut |chunk: &[f32]| {
        pcm.extend_from_slice(chunk);
        Ok(())
    },
)?;
```

`examples/synthesize.rs` est la version complète, qui écrit un WAV. Elle se lance avec :

```sh
cargo run --release --example synthesize -- "Hello." out.wav
```

Les poids descendent du Hub au premier appel et sont mis en cache, donc le premier lancement est
long et demande le réseau.

## Les points qui comptent

**La génération est un callback, pas un buffer.** `generate_audio_stream` appelle `on_chunk` à
chaque tranche décodée, à 12,5 Hz de trames. C'est ce qui permet de jouer ou d'envoyer le son
pendant qu'il se fabrique au lieu d'attendre la fin. Rendre une `Err` depuis le callback arrête la
génération et remonte.

**Le taux d'échantillonnage vient du modèle**, pas d'une constante : `model.sample_rate()`
(24000 pour les modèles actuels). Les échantillons sont des `f32` mono.

**Deux backends.** `TtsModel<F32>` est le f32 ; `TtsModel<Int8>` est la version quantifiée — moins
de mémoire, un son légèrement différent. Les deux sont des alias de types `xn`, exposés pour que
vous n'ayez pas à déclarer `xn` vous-même. Le modèle étant générique, choisir à l'exécution veut
dire une enum sur les deux variantes, comme le fait `kyutai.rs` dans ARIA.

**Une voix est un état, pas un fichier.** `get_state_for_audio_prompt` accepte un nom du catalogue
(`"alba"`, `"marius"`…), un fichier audio, ou un `.safetensors` exporté, en chemin local, `http(s)://`
ou `hf://`. Le résultat est l'état du modèle après le prompt, qu'on réutilise pour toutes les
phrases de cette voix : c'est l'étape coûteuse, gardez-la.

## Licences des voix — à lire avant de livrer

**Les 27 voix prédéfinies n'ont pas toutes la même licence, et deux interdisent l'usage
commercial.** Le crate ne redistribue pas les voix — leurs embeddings descendent du Hub à la
première utilisation — donc l'obligation pèse sur **vous**, qui diffusez l'audio. Le module
`voices` est là pour que vous puissiez la respecter sans deviner.

| Licence | Voix |
| --- | --- |
| `CC0-1.0` | 7 — `marius`, `javert`, `estelle`, `bill_boerst`, `caro_davy`, `peter_yearsley`, `stuart_bell` |
| `CC-BY-4.0` | 13 — `alba` et les 12 voix VCTK (`anna`, `vera`, `fantine`, `charles`, `paul`, `eponine`, `azelma`, `george`, `mary`, `jane`, `michael`, `eve`) |
| `CC-BY-NC-4.0` | **2 — `cosette`, `jean`. Usage non commercial uniquement.** |
| `UNKNOWN` | **5 — `giovanni`, `lola`, `juergen`, `rafael`, `daan`.** Sources dans un dépôt fermé, couvertes par aucune déclaration que nous ayons trouvée |

```rust
use pocket_tts::voices::{self, CommercialUse};

// Ce qu'un produit commercial peut utiliser. Exclut aussi les `UNKNOWN` : une licence
// indéterminée n'est pas une autorisation.
for voice in voices::commercially_usable() {
    println!("{} ({})", voice.name, voice.license);
}

// Le bloc de crédits, à coller dans un écran « À propos ».
println!("{}", voices::attribution_notice(&["alba", "marius"])?);
// Voice "alba" — Voice-acted by Alba MacKenna (CC-BY-4.0)
// Voice "marius" — Unmute Voice Donation Project — https://unmute.sh/voice-donation (CC0-1.0)

// Ou l'interrogation voix par voix.
let v = voices::find("cosette").unwrap();
assert_eq!(v.commercial_use(), CommercialUse::Forbidden);
```

`commercial_use()` renvoie trois états et non un booléen, délibérément : `Undetermined` ne doit pas
se lire comme une permission.

### Étendre ou remplacer le catalogue

Le catalogue est de la **donnée**, pas du code : un fichier JSONL par licence dans
`assets/voices/`, embarqué pour fonctionner hors ligne. Une voix de plus ne demande donc pas de
recompiler la bibliothèque :

```rust
let mine = voices::load_dir(Path::new("/chemin/vers/mes-voix"))?;
```

Chaque ligne porte `name`, `embedding` (où `{language}` est substitué à la résolution), `license`,
`attribution`, `source` et `note`. Les champs inconnus sont ignorés, donc un catalogue peut en
porter plus que cette version n'en lit. Les fichiers sont lus dans l'ordre trié, et une voix
définie deux fois est une erreur plutôt qu'un masquage silencieux.

**Le cache du Hub se pose une fois par processus**, avant tout chargement :

```rust
pocket_tts::download::set_cache_dir(dir);
```

Premier appelant gagne (`OnceLock`), donc deux appels concurrents ne se disputent pas un dossier à
mi-parcours.

**Les erreurs sont des `anyhow::Error`.** Bonnes à afficher, pas à discriminer : « poids absents »,
« réseau coupé », « config invalide » et « ce modèle exige SentencePiece » sont le même type, et il
faudrait filtrer sur le texte pour les distinguer.

C'est assumé pour la 0.1 et non définitif. Un enum concret est le bon contrat, mais le dessiner
maintenant reviendrait à deviner les variantes ; il sera dérivé des cas où les premiers
consommateurs branchent réellement. Le `0.x` est fait pour ça — une rupture y passe par un mineur.
Si vous avez besoin de discriminer une erreur, dites laquelle plutôt que de parser le message.

**`-C target-cpu=x86-64-v3` ne s'hérite pas.** Le `.cargo/config.toml` de ce crate ne vaut que pour
le dossier où cargo tourne. Sans le poser de votre côté, `xn` retombe sur du SIMD de base, plus
lent. Même chose pour `[profile.release]` : seul le profil du crate racine s'applique.

**Un seul texte à la fois.** La taille de lot est 1 et le code n'est pas pensé pour être partagé
entre threads ; une instance par thread de synthèse.

## Les quatre petits modèles anglais demandent une feature

Les 27 configs embarquées se traduisent et sont supportées, y compris les quatre à table de texte
de rang réduit : `english_2m_2026-10`, `english_2m_refl_2026-10`, `english_7m_2026-10`,
`english_7m_refl_2026-10`. Le rang voyage comme `lut_dim` et la projection est renommée au passage —
les commentaires de `Config::to_ptts` dans `src/config.rs` et de `remap_key` dans `src/weights.rs`
expliquent pourquoi ça suffit.

Deux réserves sur ces quatre-là :

- elles demandent un tokenizer **SentencePiece**, donc la feature `sentencepiece` — et avec elle
  CMake et un compilateur C++. Les autres (`english`, `english_2026-10`, `english_18m`, `french`…)
  utilisent `tokenizers`, en Rust pur : rien à installer ;
- **elles n'ont pas encore été entendues.** La traduction de config et le remappage des poids sont
  couverts par des tests unitaires, mais aucun checkpoint de rang réduit n'a été chargé ni
  généré sur une machine équipée de CMake. À considérer comme non testées, pas comme sûres.
