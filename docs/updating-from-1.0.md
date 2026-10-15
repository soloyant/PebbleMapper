# Updating from PebbleMapper 1.0.0

PebbleMapper 1.0.0 cannot tell you that a newer version exists, and cannot update itself.
Versions from 1.1 onwards can: they check once a day and update from the app. Moving from
1.0.0 to the newest version is therefore done once, by hand, as below. It takes about ten
minutes, and nothing large needs to be downloaded again.

**Which version do I have?** From 1.1 onwards the version is written in the grey bar at the
top of the page, after the name (for example *PebbleMapper version 1.1.0*). If nothing is
written there, you have 1.0.0.

## What is kept

Your projects and results (`datasets\`), the Mask R-CNN weights (`model_weights\`), the
detection models you added and your own files stay where they are, untouched. Only the
files that came with PebbleMapper 1.0.0 are replaced. If you had edited one of them (an
example file, a document), the new version's file takes its place and yours is kept beside
it with `.mine` added to its name.

## Steps

1. **Close PebbleMapper** if it is running (the **Stop PebbleMapper** button in the left
   panel, or close its black window).
2. **Download the newest release.** Open
   <https://github.com/soloyant/PebbleMapper/releases>, and under the newest release open
   **Assets** and click `pebblemapper-<version>.zip` (about 90 MB). Take this file, not
   the green **Code → Download ZIP** button: only the release file can update itself later.
3. **Extract it to a new folder**, for example `C:\PebbleMapper-new`: right-click the file
   in the Downloads folder, **Extract All…**, **Browse…**, **Extract**. Do not extract it
   over your current PebbleMapper folder.
4. **Run the installer** in that new folder: double-click **Install PebbleMapper**. (If
   Windows warns that the file comes from the internet: **More info**, then **Run anyway**.)
5. The installer finds your current PebbleMapper through its desktop icon and asks:

   ```
   PebbleMapper is already installed in C:\PebbleMapper
   Your projects, results and model weights are there.
   Update that installation to this version, keeping all of them? [Y/n]
   ```

   Press **Enter**. It updates that folder, reuses the software already installed, and ends
   with `PebbleMapper is installed.`
6. **Start PebbleMapper** from its desktop icon as usual. The grey bar now shows the new
   version. The folder you extracted to in step 3 is no longer needed: delete it.

From now on, a newer version is announced under the top bar and on the **Models** tab, and
**Update** installs it without any of the steps above.

## If the installer does not ask

The installer finds PebbleMapper 1.0.0 through the **PebbleMapper** icon on the desktop. If
that icon was deleted or moved, it installs the new version separately in the new folder
instead. Two ways to continue:

- **Point it at your PebbleMapper folder.** Open the new folder in File Explorer, click in
  the address bar, type `cmd` and press Enter. In the black window, type the line below with
  your own PebbleMapper folder (the one that holds `datasets` and `model_weights`), then
  press Enter:

  ```
  "Install PebbleMapper.bat" -UpdateFolder "C:\PebbleMapper"
  ```

- **Or move your work into the new installation.** Leave the new version where the
  installer put it. In File Explorer, move your project folders from the old
  `datasets\` folder into the new one (every folder except `example_…`), and
  `mask_rcnn_clasts.h5` from the old `model_weights\` folder into the new one. Start the
  new version from the desktop icon (the installer created it). Once your projects open
  there, delete the old folder.

## Hearing about the next versions

From 1.1, PebbleMapper itself tells you. To also receive an e-mail when a version is
published, sign in to GitHub, open <https://github.com/soloyant/PebbleMapper>, click
**Watch → Custom**, tick **Releases** and **Apply**.

## Something went wrong

The installer never leaves your PebbleMapper half-updated: if it stops, the folder is left
as it was, and its last lines say why. For anything else, open an issue at
<https://github.com/soloyant/PebbleMapper/issues> (a free GitHub account is needed) and
attach the diagnostic bundle (**Build diagnostic bundle** in the left panel).
