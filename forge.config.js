const { FusesPlugin } = require('@electron-forge/plugin-fuses');
const { FuseV1Options, FuseVersion } = require('@electron/fuses');

module.exports = {
    packagerConfig: {
        asar: {
            unpack: '**/{onnxruntime-node,@ricky0123}/**/*',
        },
        extraResource: ['./src/assets/SystemAudioDump'],
        name: 'Cheating Daddy',
        icon: 'src/assets/logo',
        // use `security find-identity -v -p codesigning` to find your identity
        // for macos signing
        // also fuck apple
        // osxSign: {
        //    identity: '<paste your identity here>',
        //   optionsForFile: (filePath) => {
        //       return {
        //           entitlements: 'entitlements.plist',
        //       };
        //   },
        // },
        // notarize if off cuz i ran this for 6 hours and it still didnt finish
        // osxNotarize: {
        //    appleId: 'your apple id',
        //    appleIdPassword: 'app specific password',
        //    teamId: 'your team id',
        // },
    },
    rebuildConfig: {},
    hooks: {
        // After packaging: set permissions and ad-hoc sign the app for macOS
        postPackage: async (forgeConfig, options) => {
            if (options.platform === 'darwin') {
                const fs = require('fs');
                const path = require('path');
                const { execSync } = require('child_process');

                for (const outputPath of options.outputPaths) {
                    const appPath = path.join(outputPath, 'Cheating Daddy.app');
                    const binaryPath = path.join(appPath, 'Contents', 'Resources', 'SystemAudioDump');

                    // 1. Set execute permissions for SystemAudioDump
                    try {
                        fs.chmodSync(binaryPath, 0o755);
                        console.log('Set execute permissions for SystemAudioDump');
                    } catch (error) {
                        console.warn('Failed to set permissions for SystemAudioDump:', error.message);
                    }

                    // 2. Ad-hoc sign the entire .app bundle (including SystemAudioDump)
                    // This is required for macOS TCC to properly grant screen recording
                    // and system audio capture permissions to the app and its helper binaries.
                    // Without signing, SystemAudioDump can't capture audio when the app
                    // is launched via Finder (double-click) instead of Terminal.
                    try {
                        console.log('Ad-hoc signing app bundle...');
                        execSync(`codesign --force --deep --sign - "${appPath}"`, { stdio: 'pipe' });
                        console.log('Ad-hoc signed successfully:', appPath);
                    } catch (error) {
                        console.warn('Ad-hoc signing failed (app may not work when launched from Finder):', error.message);
                    }
                }
            }
        },
    },
    makers: [
        {
            name: '@electron-forge/maker-squirrel',
            config: {
                name: 'cheating-daddy',
                productName: 'Cheating Daddy Pro',
                shortcutName: 'Cheating Daddy Pro',
                createDesktopShortcut: false,
                createStartMenuShortcut: true,
            },
        },
        {
            name: '@electron-forge/maker-dmg',
            platforms: ['darwin'],
        },
        {
            name: '@reforged/maker-appimage',
            platforms: ['linux'],
            config: {
                options: {
                    name: 'Cheating Daddy',
                    productName: 'Cheating Daddy Pro',
                    genericName: 'AI Assistant',
                    description: 'AI assistant for interviews and learning',
                    categories: ['Development', 'Education'],
                    icon: 'src/assets/logo',
                    desktopIntegration: false  // Prevent automatic desktop shortcut creation
                }
            },
        },
    ],
    plugins: [
        {
            name: '@electron-forge/plugin-auto-unpack-natives',
            config: {},
        },
        // Fuses are used to enable/disable various Electron functionality
        // at package time, before code signing the application
        new FusesPlugin({
            version: FuseVersion.V1,
            [FuseV1Options.RunAsNode]: false,
            [FuseV1Options.EnableCookieEncryption]: true,
            [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
            [FuseV1Options.EnableNodeCliInspectArguments]: false,
            [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
            [FuseV1Options.OnlyLoadAppFromAsar]: true,
        }),
    ],
};
