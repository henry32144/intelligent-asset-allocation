import React from 'react';
import Button from '@material-ui/core/Button';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import Grow from '@material-ui/core/Grow';
import Paper from '@material-ui/core/Paper';
import Popper from '@material-ui/core/Popper';
import MenuItem from '@material-ui/core/MenuItem';
import MenuList from '@material-ui/core/MenuList';
import { makeStyles } from '@material-ui/core/styles';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import AddIcon from '@material-ui/icons/Add';
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import Divider from '@material-ui/core/Divider';

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
  },
  expandButton: {
    color: 'inherit'
  },
  popperRoot: {
    zIndex: 1400,
  },
  popperMenuItem: {
    paddingRight: "8px"
  },
}));

export default function PortfolioMenuButtons(props) {
  const classes = useStyles();
  const [open, setOpen] = React.useState(false);
  const anchorRef = React.useRef(null);

  const toogleDrawer = () => {
    props.setSideBarExpand(!props.isSideBarExpanded);
  };

  const handlePortfolioToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }
    setOpen(false);
  };

  const portfolioButtonOnClick = (e, id) => {
    props.setCurrentSelectedPortfolio(id);
    console.log(id);
    var portfolioStocks = props.userPortfolios.find(function(item, index, array){
      return item.portfolioId === id;
    }).portfolioStocks;
    props.setSelectedStocks(portfolioStocks);
    props.setSideBarExpand(true);
    handleClose(e)
  }

  const createPortfolioOnClick = (e) => {
    if (props.userData.userId == undefined) {
      props.setDialogTitle("Error")
      props.setDialogMessage("Please login first");
      props.openMessageDialog();
    } else {
      props.handleCreatePortfolioDialogOpen();
    }
    handleClose(e)
  };

  const handleListKeyDown = (event) => {
    if (event.key === 'Tab') {
      event.preventDefault();
      setOpen(false);
    }
  }

  // return focus to the button when we transitioned from !open -> open
  const prevOpen = React.useRef(open);
  React.useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current.focus();
    }
    prevOpen.current = open;
  }, [open]);


  const portfolioMenuItems = props.userPortfolios.map((portfolio) =>
    <MenuItem key={portfolio.portfolioId.toString()} onClick={(e) => {portfolioButtonOnClick(e, portfolio.portfolioId)}}>
      <Typography variant="inherit" noWrap>
        {portfolio.portfolioName}
      </Typography>
    </MenuItem>
  );

  return (
    <div className={classes.root}>
      <IconButton
        edge="start"
        className={classes.menuButton}
        onClick={toogleDrawer}
        color="inherit"
        aria-label="menu">
        <MenuIcon />
      </IconButton>
      <Button
        endIcon={<ExpandMoreIcon />}
        ref={anchorRef}
        aria-controls={open ? 'menu-list-grow' : undefined}
        aria-haspopup="true"
        className={classes.expandButton}
        onClick={handlePortfolioToggle}
      >
        <Typography variant="inherit" noWrap>
          { props.currentSelectedPortfolio == undefined || props.userPortfolios.length == 0 ?
            "My Portfolios"
            :
            props.userPortfolios.find(function(item, index, array){
              return item.portfolioId === props.currentSelectedPortfolio;
            }).portfolioName
          }
        </Typography>
      </Button>
      <Popper
        className={classes.popperRoot}
        open={open}
        anchorEl={anchorRef.current}
        role={undefined}
        transition
        disablePortal>
        {({ TransitionProps, placement }) => (
          <Grow
            {...TransitionProps}
            style={{ transformOrigin: placement === 'bottom' ? 'center top' : 'center bottom' }}
          >
            <Paper>
              <ClickAwayListener onClickAway={handleClose}>
                <MenuList autoFocusItem={open} id="menu-list-grow" onKeyDown={handleListKeyDown}>
                  {portfolioMenuItems}
                  <Divider light component="li" />
                  <MenuItem onClick={createPortfolioOnClick}>
                    <Typography variant="inherit" noWrap className={classes.popperMenuItem}>
                      Create New Portfolio
                      </Typography>
                    <AddIcon fontSize="small" className={classes.popperMenuItem} />
                  </MenuItem>
                </MenuList>
              </ClickAwayListener>
            </Paper>
          </Grow>
        )}
      </Popper>
    </div>
  );
}
